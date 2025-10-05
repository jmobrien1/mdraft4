#!/usr/bin/env python3
"""
RFP Requirements Extraction System - Standalone Version
Single file with built-in web UI and SQLite database
"""

import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from dataclasses import dataclass

# Configuration
DB_FILE = "rfp_extraction.db"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@dataclass
class Requirement:
    """Data model for a requirement"""
    id: str
    document_id: str
    original_text: str
    clean_text: str
    classification: str
    source_section: str
    source_subsection: str
    confidence_score: float
    status: str = "ai_extracted"
    validated_by: Optional[str] = None


class Database:
    """Simple SQLite database handler"""

    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_size INTEGER,
                mime_type TEXT,
                status TEXT DEFAULT 'uploaded',
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_started_at TIMESTAMP,
                processing_completed_at TIMESTAMP
            )
        """)

        # Requirements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS requirements (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                original_text TEXT NOT NULL,
                clean_text TEXT NOT NULL,
                classification TEXT NOT NULL,
                source_section TEXT,
                source_subsection TEXT,
                confidence_score REAL,
                status TEXT DEFAULT 'ai_extracted',
                validated_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)

        conn.commit()
        conn.close()

    def save_document(self, doc_id: str, filename: str, stored_filename: str, file_size: int, mime_type: str):
        """Save document metadata"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (id, original_filename, stored_filename, file_size, mime_type)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, filename, stored_filename, file_size, mime_type))
        conn.commit()
        conn.close()

    def save_requirements(self, requirements: List[Requirement]):
        """Save multiple requirements"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        for req in requirements:
            cursor.execute("""
                INSERT INTO requirements
                (id, document_id, original_text, clean_text, classification,
                 source_section, source_subsection, confidence_score, status, validated_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                req.id, req.document_id, req.original_text, req.clean_text,
                req.classification, req.source_section, req.source_subsection,
                req.confidence_score, req.status, req.validated_by
            ))

        conn.commit()
        conn.close()

    def get_documents(self) -> List[Dict]:
        """Get all documents"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
        docs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return docs

    def get_requirements(self, document_id: str) -> List[Dict]:
        """Get requirements for a document"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM requirements
            WHERE document_id = ?
            ORDER BY source_subsection, classification
        """, (document_id,))
        reqs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return reqs

    def update_requirement_status(self, req_id: str, status: str, validated_by: Optional[str] = None):
        """Update requirement validation status"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE requirements
            SET status = ?, validated_by = ?
            WHERE id = ?
        """, (status, validated_by, req_id))
        conn.commit()
        conn.close()

    def get_stats(self, document_id: str) -> Dict:
        """Get statistics for a document"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN classification = 'PERFORMANCE_REQUIREMENT' THEN 1 ELSE 0 END) as performance,
                SUM(CASE WHEN classification = 'COMPLIANCE_REQUIREMENT' THEN 1 ELSE 0 END) as compliance,
                SUM(CASE WHEN classification = 'DELIVERABLE_REQUIREMENT' THEN 1 ELSE 0 END) as deliverable,
                SUM(CASE WHEN status = 'human_validated' THEN 1 ELSE 0 END) as validated
            FROM requirements
            WHERE document_id = ?
        """, (document_id,))

        row = cursor.fetchone()
        conn.close()

        return {
            "total": row[0],
            "performance": row[1],
            "compliance": row[2],
            "deliverable": row[3],
            "validated": row[4]
        }


class RequirementExtractor:
    """Extract and classify requirements from RFP text"""

    # Classification patterns
    PATTERNS = {
        "PERFORMANCE_REQUIREMENT": [
            r"shall\s+(?:maintain|achieve|ensure|provide|support)\s+.*?(?:\d+%|\d+\s+(?:seconds|minutes|hours|days))",
            r"uptime.*?\d+%",
            r"response\s+time.*?\d+\s+(?:seconds|milliseconds)",
            r"availability.*?\d+%",
            r"processing.*?within\s+\d+",
            r"latency.*?\d+",
        ],
        "COMPLIANCE_REQUIREMENT": [
            r"shall\s+comply\s+with",
            r"must\s+(?:meet|satisfy|adhere\s+to)",
            r"in\s+accordance\s+with",
            r"(?:FISMA|NIST|ISO|SOC|HIPAA|FedRAMP)",
            r"encryption.*?(?:AES|TLS|SSL)",
            r"security\s+(?:standards|requirements|controls)",
            r"audit.*?requirements",
            r"(?:authentication|authorization).*?(?:MFA|multi-factor)",
        ],
        "DELIVERABLE_REQUIREMENT": [
            r"shall\s+(?:submit|provide|deliver|furnish)",
            r"(?:report|documentation|deliverable).*?(?:monthly|weekly|quarterly|annually)",
            r"contractor\s+shall\s+prepare",
            r"(?:plan|document|report).*?shall\s+be\s+(?:submitted|provided|delivered)",
            r"by\s+the\s+\d+(?:st|nd|rd|th)\s+(?:day|business\s+day)",
        ],
    }

    def extract_requirements(self, text: str, document_id: str) -> List[Requirement]:
        """Extract requirements from text"""
        requirements = []

        # Split into sections and process
        lines = text.split('\n')
        current_section = "Unknown"
        current_subsection = "0"

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            section_match = re.match(r'^(Section\s+[\w.]+)[\s:]+(.+)', line, re.IGNORECASE)
            if section_match:
                current_section = section_match.group(1)
                current_subsection = str(line_num)
                continue

            # Check if line is a requirement (contains "shall" or "must")
            if not re.search(r'\b(?:shall|must|will\s+be\s+required\s+to)\b', line, re.IGNORECASE):
                continue

            # Classify the requirement
            classification, confidence = self._classify_requirement(line)

            if classification:
                req = Requirement(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    original_text=line,
                    clean_text=self._clean_text(line),
                    classification=classification,
                    source_section=current_section,
                    source_subsection=current_subsection,
                    confidence_score=confidence,
                    status="ai_extracted"
                )
                requirements.append(req)

        return requirements

    def _classify_requirement(self, text: str) -> tuple[Optional[str], float]:
        """Classify a requirement and return confidence score"""
        scores = {}

        for classification, patterns in self.PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1

            if score > 0:
                scores[classification] = score / len(patterns)

        if not scores:
            return None, 0.0

        best_classification = max(scores, key=scores.get)
        confidence = scores[best_classification]

        return best_classification, confidence

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters at start/end
        text = text.strip('•-–—*\t ')
        return text


# ===== STREAMLIT UI =====

def main():
    st.set_page_config(
        page_title="RFP Requirements Extractor",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 RFP Requirements Extraction System")
    st.markdown("Upload RFP documents to automatically extract and classify requirements")

    # Initialize database
    db = Database()
    extractor = RequirementExtractor()

    # Sidebar
    with st.sidebar:
        st.header("📊 Quick Stats")
        docs = db.get_documents()
        st.metric("Total Documents", len(docs))

        total_reqs = sum([db.get_stats(doc['id'])['total'] for doc in docs])
        st.metric("Total Requirements", total_reqs)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📋 Review", "📈 Analytics"])

    with tab1:
        upload_tab(db, extractor)

    with tab2:
        review_tab(db)

    with tab3:
        analytics_tab(db)


def upload_tab(db: Database, extractor: RequirementExtractor):
    """Upload and process documents"""
    st.header("Upload RFP Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'docx'],
        help="Upload RFP documents in TXT, PDF, or DOCX format"
    )

    if uploaded_file is not None:
        # Save file
        doc_id = str(uuid.uuid4())
        file_extension = Path(uploaded_file.name).suffix
        stored_filename = f"{doc_id}{file_extension}"
        file_path = UPLOAD_DIR / stored_filename

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Save to database
        db.save_document(
            doc_id=doc_id,
            filename=uploaded_file.name,
            stored_filename=stored_filename,
            file_size=uploaded_file.size,
            mime_type=uploaded_file.type or "application/octet-stream"
        )

        st.success(f"✅ Uploaded: {uploaded_file.name}")

        # Process file
        with st.spinner("🔄 Extracting requirements..."):
            # Read file content
            if file_extension == '.txt':
                text = uploaded_file.getvalue().decode('utf-8')
            else:
                st.warning("⚠️ PDF/DOCX parsing not included in standalone version. Use .txt files.")
                return

            # Extract requirements
            requirements = extractor.extract_requirements(text, doc_id)

            # Save requirements
            if requirements:
                db.save_requirements(requirements)
                st.success(f"✅ Extracted {len(requirements)} requirements!")

                # Show preview
                st.subheader("Preview")
                for req in requirements[:5]:
                    with st.expander(f"{req.classification} (Confidence: {req.confidence_score:.0%})"):
                        st.write(f"**Text:** {req.clean_text}")
                        st.write(f"**Source:** {req.source_section}.{req.source_subsection}")

                if len(requirements) > 5:
                    st.info(f"Showing 5 of {len(requirements)} requirements. View all in the Review tab.")
            else:
                st.warning("No requirements found. Make sure the document contains 'shall' or 'must' statements.")


def review_tab(db: Database):
    """Review and validate requirements"""
    st.header("Review Requirements")

    docs = db.get_documents()

    if not docs:
        st.info("No documents uploaded yet. Go to the Upload tab to get started.")
        return

    # Document selector
    doc_names = {doc['original_filename']: doc['id'] for doc in docs}
    selected_doc_name = st.selectbox("Select Document", list(doc_names.keys()))
    selected_doc_id = doc_names[selected_doc_name]

    # Get requirements
    requirements = db.get_requirements(selected_doc_id)

    if not requirements:
        st.warning("No requirements found for this document.")
        return

    # Filter by classification
    classifications = list(set([r['classification'] for r in requirements]))
    filter_class = st.multiselect(
        "Filter by Type",
        classifications,
        default=classifications
    )

    filtered_reqs = [r for r in requirements if r['classification'] in filter_class]

    st.write(f"Showing {len(filtered_reqs)} of {len(requirements)} requirements")

    # Display requirements
    for req in filtered_reqs:
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            status_icon = "✅" if req['status'] == 'human_validated' else "⏳"
            st.write(f"{status_icon} **{req['classification'].replace('_', ' ').title()}**")
            st.write(req['clean_text'])
            st.caption(f"Source: {req['source_section']}.{req['source_subsection']} | Confidence: {req['confidence_score']:.0%}")

        with col2:
            if st.button("✅ Validate", key=f"validate_{req['id']}"):
                db.update_requirement_status(req['id'], 'human_validated', 'user@example.com')
                st.rerun()

        with col3:
            if st.button("❌ Reject", key=f"reject_{req['id']}"):
                db.update_requirement_status(req['id'], 'rejected', 'user@example.com')
                st.rerun()

        st.divider()


def analytics_tab(db: Database):
    """Show analytics and statistics"""
    st.header("Analytics Dashboard")

    docs = db.get_documents()

    if not docs:
        st.info("No documents uploaded yet.")
        return

    # Document selector
    doc_names = {doc['original_filename']: doc['id'] for doc in docs}
    selected_doc_name = st.selectbox("Select Document", list(doc_names.keys()), key="analytics_doc")
    selected_doc_id = doc_names[selected_doc_name]

    # Get stats
    stats = db.get_stats(selected_doc_id)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Requirements", stats['total'])

    with col2:
        st.metric("Performance", stats['performance'])

    with col3:
        st.metric("Compliance", stats['compliance'])

    with col4:
        st.metric("Deliverable", stats['deliverable'])

    # Validation progress
    st.subheader("Validation Progress")
    if stats['total'] > 0:
        progress = stats['validated'] / stats['total']
        st.progress(progress)
        st.write(f"{stats['validated']} of {stats['total']} validated ({progress:.0%})")
    else:
        st.write("No requirements to validate")

    # Requirements by type
    st.subheader("Requirements by Type")
    chart_data = {
        "Performance": stats['performance'],
        "Compliance": stats['compliance'],
        "Deliverable": stats['deliverable']
    }
    st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
