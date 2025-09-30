"""
Streamlit frontend for RFP Extraction and Analysis Platform.

This is the "Correction Console" interface for human validation of AI-extracted
requirements. It provides an efficient workflow for proposal managers to review
and validate extracted data with full audit trail support.
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_DIR = "uploads"

# Page configuration
st.set_page_config(
    page_title="RFP Extraction Platform",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .requirement-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-ai-extracted {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .status-human-validated {
        background-color: #e8f5e8;
        color: #2e7d32;
    }
    .status-human-corrected {
        background-color: #fff3e0;
        color: #f57c00;
    }
</style>
""", unsafe_allow_html=True)

def get_api_data(endpoint: str, params: Dict = None) -> Optional[Dict]:
    """Helper function to get data from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def post_api_data(endpoint: str, data: Dict) -> Optional[Dict]:
    """Helper function to post data to the API"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def put_api_data(endpoint: str, data: Dict) -> Optional[Dict]:
    """Helper function to put data to the API"""
    try:
        response = requests.put(f"{API_BASE_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def get_confidence_class(confidence: str) -> str:
    """Get CSS class for confidence level"""
    if confidence == "high":
        return "confidence-high"
    elif confidence == "medium":
        return "confidence-medium"
    elif confidence == "low":
        return "confidence-low"
    return ""

def get_status_class(status: str) -> str:
    """Get CSS class for status"""
    return f"status-{status.replace('_', '-')}"

def display_requirement_card(requirement: Dict, index: int):
    """Display a requirement card with validation interface"""
    with st.container():
        st.markdown(f"""
        <div class="requirement-card">
            <h4>Requirement #{index + 1}</h4>
            <p><strong>Classification:</strong> {requirement['classification']}</p>
            <p><strong>Confidence:</strong> <span class="{get_confidence_class(requirement.get('ai_confidence_score', ''))}">{requirement.get('ai_confidence_score', 'Unknown')}</span></p>
            <p><strong>Source:</strong> Page {requirement['source_page']}, Section {requirement.get('source_section', 'Unknown')}</p>
            <p><strong>Status:</strong> <span class="status-badge {get_status_class(requirement['status'])}">{requirement['status']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Raw text display
        st.markdown("**Raw Text:**")
        st.text_area(
            "Raw Text",
            value=requirement['raw_text'],
            height=100,
            key=f"raw_text_{requirement['id']}",
            disabled=True
        )
        
        # Clean text editing
        st.markdown("**Clean Text:**")
        clean_text = st.text_area(
            "Clean Text",
            value=requirement.get('clean_text', ''),
            height=100,
            key=f"clean_text_{requirement['id']}"
        )
        
        # Classification selection
        classification_options = [
            "PERFORMANCE_REQUIREMENT",
            "DELIVERABLE_REQUIREMENT", 
            "COMPLIANCE_REQUIREMENT",
            "EVALUATION_CRITERIA",
            "INSTRUCTION",
            "FAR_CLAUSE",
            "OTHER"
        ]
        
        current_classification = requirement['classification']
        classification_index = classification_options.index(current_classification) if current_classification in classification_options else 0
        
        new_classification = st.selectbox(
            "Classification",
            options=classification_options,
            index=classification_index,
            key=f"classification_{requirement['id']}"
        )
        
        # Validation notes
        validation_notes = st.text_area(
            "Validation Notes",
            value="",
            height=60,
            key=f"notes_{requirement['id']}",
            placeholder="Add any notes about this requirement..."
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Approve", key=f"approve_{requirement['id']}"):
                validation_data = {
                    "action": "approve",
                    "clean_text": clean_text,
                    "classification": new_classification,
                    "validation_notes": validation_notes
                }
                
                result = put_api_data(f"/requirements/{requirement['id']}", validation_data)
                if result:
                    st.success("Requirement approved successfully!")
                    st.rerun()
        
        with col2:
            if st.button("✏️ Correct & Approve", key=f"correct_{requirement['id']}"):
                validation_data = {
                    "action": "correct",
                    "clean_text": clean_text,
                    "classification": new_classification,
                    "validation_notes": validation_notes
                }
                
                result = put_api_data(f"/requirements/{requirement['id']}", validation_data)
                if result:
                    st.success("Requirement corrected and approved!")
                    st.rerun()
        
        with col3:
            if st.button("🚩 Flag for Review", key=f"flag_{requirement['id']}"):
                validation_data = {
                    "action": "flag",
                    "clean_text": clean_text,
                    "classification": new_classification,
                    "validation_notes": validation_notes
                }
                
                result = put_api_data(f"/requirements/{requirement['id']}", validation_data)
                if result:
                    st.warning("Requirement flagged for review!")
                    st.rerun()
        
        st.markdown("---")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📋 RFP Extraction Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Human-in-the-Loop Validation Console**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # System health check
        st.subheader("System Status")
        health_data = get_api_data("/health")
        if health_data:
            if health_data["status"] == "healthy":
                st.success("✅ System Healthy")
            else:
                st.error("❌ System Issues")
                st.error(health_data.get("error", "Unknown error"))
        
        # Statistics
        st.subheader("📊 Statistics")
        stats_data = get_api_data("/stats")
        if stats_data:
            st.metric("Total Documents", stats_data["total_documents"])
            st.metric("Total Requirements", stats_data["total_requirements"])
            st.metric("Total Chunks", stats_data["total_chunks"])
        
        # Filters
        st.subheader("🔍 Filters")
        confidence_filter = st.selectbox(
            "Confidence Level",
            ["All", "low", "medium", "high"],
            key="confidence_filter"
        )
        
        classification_filter = st.selectbox(
            "Classification",
            ["All", "PERFORMANCE_REQUIREMENT", "DELIVERABLE_REQUIREMENT", 
             "COMPLIANCE_REQUIREMENT", "EVALUATION_CRITERIA", "INSTRUCTION", "FAR_CLAUSE", "OTHER"],
            key="classification_filter"
        )
        
        # Pagination
        st.subheader("📄 Pagination")
        page_size = st.slider("Items per page", 5, 50, 20)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Review Queue", "📁 Documents", "🔍 Search", "📊 Analytics"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Review Queue</h2>', unsafe_allow_html=True)
        
        # Get review queue data
        params = {
            "limit": page_size,
            "skip": 0
        }
        
        if confidence_filter != "All":
            params["confidence_filter"] = confidence_filter
        
        if classification_filter != "All":
            params["classification_filter"] = classification_filter
        
        review_data = get_api_data("/requirements/review_queue", params)
        
        if review_data and review_data["items"]:
            st.info(f"Showing {len(review_data['items'])} of {review_data['total_count']} requirements pending validation")
            
            for i, requirement in enumerate(review_data["items"]):
                display_requirement_card(requirement, i)
        else:
            st.info("No requirements pending validation. Great job! 🎉")
    
    with tab2:
        st.markdown('<h2 class="section-header">Documents</h2>', unsafe_allow_html=True)
        
        # Document upload
        st.subheader("📤 Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx'],
            help="Upload RFP documents for processing"
        )
        
        if uploaded_file is not None:
            if st.button("Upload Document"):
                # Save file temporarily
                file_path = f"{UPLOAD_DIR}/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Upload to API
                with open(file_path, "rb") as f:
                    files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                    try:
                        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
                        response.raise_for_status()
                        result = response.json()
                        st.success(f"Document uploaded successfully! Job ID: {result['job_id']}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Upload failed: {e}")
        
        # List documents
        st.subheader("📋 Document List")
        documents_data = get_api_data("/documents")
        
        if documents_data:
            for doc in documents_data:
                with st.expander(f"📄 {doc['original_filename']} - {doc['status']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**File Type:** {doc['file_type']}")
                        st.write(f"**Size:** {doc['file_size']:,} bytes")
                    with col2:
                        st.write(f"**Status:** {doc['status']}")
                        st.write(f"**Uploaded:** {doc['uploaded_at']}")
                    with col3:
                        st.write(f"**Engine:** {doc.get('processing_engine_used', 'N/A')}")
                        st.write(f"**Confidence:** {doc.get('extraction_confidence', 'N/A')}")
        else:
            st.info("No documents uploaded yet.")
    
    with tab3:
        st.markdown('<h2 class="section-header">Search Requirements</h2>', unsafe_allow_html=True)
        
        search_query = st.text_input("Search query", placeholder="Enter search terms...")
        
        if search_query:
            search_data = {
                "query": search_query,
                "limit": 20
            }
            
            search_results = post_api_data("/search", search_data)
            
            if search_results and search_results["results"]:
                st.info(f"Found {search_results['total_count']} results")
                
                for i, result in enumerate(search_results["results"]):
                    with st.expander(f"Result #{i+1} - {result['classification']}"):
                        st.write(f"**Source:** Page {result['source_page']}, Section {result.get('source_section', 'Unknown')}")
                        st.write(f"**Text:** {result['raw_text'][:200]}...")
                        if result.get('clean_text'):
                            st.write(f"**Clean Text:** {result['clean_text'][:200]}...")
            else:
                st.info("No results found.")
    
    with tab4:
        st.markdown('<h2 class="section-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        if stats_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Document Status")
                for status, count in stats_data["documents_by_status"].items():
                    st.metric(status.replace("_", " ").title(), count)
            
            with col2:
                st.subheader("🏷️ Classification Distribution")
                for classification, count in stats_data["requirements_by_classification"].items():
                    st.metric(classification.replace("_", " ").title(), count)
            
            with col3:
                st.subheader("✅ Validation Status")
                for status, count in stats_data["validation_status_counts"].items():
                    st.metric(status.replace("_", " ").title(), count)
            
            # System health details
            st.subheader("🔧 System Health")
            health_details = stats_data.get("system_health", {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Database Connection", "✅ Connected" if health_details.get("database_connection") else "❌ Disconnected")
            with col2:
                st.metric("Tables Exist", "✅ Yes" if health_details.get("tables_exist") else "❌ No")
            with col3:
                st.metric("pgvector Available", "✅ Yes" if health_details.get("pgvector_available") else "❌ No")

if __name__ == "__main__":
    main()

