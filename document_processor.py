"""
Document Processing Pipeline for RFP Extraction Platform

This module implements the dual-engine processing pipeline as specified in the RFP blueprint:
1. Secondary Engine: PyMuPDF + Unstructured.io (cost-effective, fast)
2. Primary Engine: Google Document AI (high-fidelity, for complex documents)
3. Intelligent chunking with RFP-aware segmentation
4. AI-powered requirement classification
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

# Document processing libraries
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import re

# AI and ML libraries
import openai
from sentence_transformers import SentenceTransformer

# Database imports
from sqlalchemy.orm import Session
from database import get_db_session
from models import Document, TextChunk, Requirement, ProcessingJob, ProcessingStatus, ValidationStatus, RequirementClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# Initialize AI models
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Initialize sentence transformer for embeddings
try:
    embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    logger.info(f"Loaded embedding model: {SENTENCE_TRANSFORMER_MODEL}")
except Exception as e:
    logger.warning(f"Could not load embedding model: {e}")
    embedding_model = None


class DocumentProcessor:
    """
    Main document processor implementing the dual-engine pipeline.
    
    This class handles the complete document processing workflow:
    1. File type detection and routing
    2. Text extraction using appropriate engine
    3. Intelligent chunking
    4. Requirement classification
    5. Database storage
    """
    
    def __init__(self):
        self.embedding_model = embedding_model
        self.openai_available = bool(OPENAI_API_KEY)
    
    async def process_document(self, document_id: uuid.UUID, job_id: uuid.UUID) -> bool:
        """
        Main document processing entry point.
        
        Args:
            document_id: UUID of the document to process
            job_id: UUID of the processing job
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            with get_db_session() as db:
                # Get document and job
                document = db.query(Document).filter(Document.id == document_id).first()
                job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
                
                if not document or not job:
                    logger.error(f"Document {document_id} or job {job_id} not found")
                    return False
                
                # Update job status
                job.status = "running"
                job.started_at = datetime.utcnow()
                db.commit()
                
                # Process the document
                success = await self._process_document_file(document, job, db)
                
                # Update job completion
                if success:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    document.status = ProcessingStatus.EXTRACTION_COMPLETE
                    document.processed_at = datetime.utcnow()
                else:
                    job.status = "failed"
                    job.error_message = "Document processing failed"
                    document.status = ProcessingStatus.ERROR
                
                db.commit()
                return success
                
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            # Update job with error
            try:
                with get_db_session() as db:
                    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
                    if job:
                        job.status = "failed"
                        job.error_message = str(e)
                        job.completed_at = datetime.utcnow()
                        db.commit()
            except:
                pass
            return False
    
    async def _process_document_file(self, document: Document, job: ProcessingJob, db: Session) -> bool:
        """
        Process a single document file.
        
        Args:
            document: Document model instance
            job: Processing job instance
            db: Database session
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            file_path = Path(document.file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Determine processing engine based on file type and complexity
            engine_used = self._select_processing_engine(document, file_path)
            document.processing_engine_used = engine_used
            
            # Extract text using selected engine
            if engine_used == "pymupdf":
                text_content = await self._extract_text_pymupdf(file_path)
            elif engine_used == "unstructured":
                text_content = await self._extract_text_unstructured(file_path)
            elif engine_used == "document_ai":
                text_content = await self._extract_text_document_ai(file_path)
            else:
                logger.error(f"Unknown processing engine: {engine_used}")
                return False
            
            if not text_content:
                logger.error("No text extracted from document")
                return False
            
            # Perform intelligent chunking
            chunks = await self._intelligent_chunking(text_content, document)
            
            # Store chunks in database
            chunk_objects = []
            for i, chunk_data in enumerate(chunks):
                chunk = TextChunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    chunk_index=i,
                    section_identifier=chunk_data.get('section'),
                    subsection_identifier=chunk_data.get('subsection'),
                    raw_text=chunk_data['text'],
                    cleaned_text=chunk_data.get('cleaned_text'),
                    source_page=chunk_data.get('page', 1),
                    source_paragraph=chunk_data.get('paragraph'),
                    chunk_type=chunk_data.get('type', 'paragraph'),
                    confidence_score=chunk_data.get('confidence', 'medium')
                )
                
                # Generate embedding if model is available
                if self.embedding_model:
                    try:
                        embedding = self.embedding_model.encode(chunk_data['text'])
                        chunk.embedding = embedding.tolist()  # Convert to list for JSON storage
                    except Exception as e:
                        logger.warning(f"Could not generate embedding for chunk {i}: {e}")
                
                chunk_objects.append(chunk)
                db.add(chunk)
            
            db.commit()
            logger.info(f"Stored {len(chunk_objects)} text chunks for document {document.id}")
            
            # Extract and classify requirements
            requirements = await self._extract_requirements(chunk_objects, document)
            
            # Store requirements in database
            for req_data in requirements:
                requirement = Requirement(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    source_chunk_id=req_data['chunk_id'],
                    raw_text=req_data['raw_text'],
                    clean_text=req_data.get('clean_text'),
                    classification=req_data['classification'],
                    source_page=req_data['source_page'],
                    source_paragraph=req_data.get('source_paragraph'),
                    source_section=req_data.get('source_section'),
                    source_subsection=req_data.get('source_subsection'),
                    ai_confidence_score=req_data.get('confidence', 'medium'),
                    extraction_method=engine_used,
                    status=ValidationStatus.AI_EXTRACTED
                )
                db.add(requirement)
            
            db.commit()
            logger.info(f"Stored {len(requirements)} requirements for document {document.id}")
            
            # Update job progress
            job.total_items = len(chunk_objects) + len(requirements)
            job.processed_items = job.total_items
            db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing document file: {e}")
            return False
    
    def _select_processing_engine(self, document: Document, file_path: Path) -> str:
        """
        Select the appropriate processing engine based on document characteristics.
        
        Args:
            document: Document model instance
            file_path: Path to the document file
            
        Returns:
            str: Engine name to use
        """
        # For now, use PyMuPDF for PDFs and basic text extraction for others
        # In production, this would include more sophisticated logic
        
        if document.file_type.lower() == 'pdf':
            # Check file size and complexity
            if document.file_size > 10 * 1024 * 1024:  # > 10MB
                return "document_ai"  # Use Document AI for large, complex PDFs
            else:
                return "pymupdf"  # Use PyMuPDF for smaller PDFs
        elif document.file_type.lower() in ['docx', 'doc']:
            return "unstructured"  # Use Unstructured.io for Word documents
        else:
            return "pymupdf"  # Default fallback
    
    async def _extract_text_pymupdf(self, file_path: Path) -> Optional[str]:
        """
        Extract text using PyMuPDF (fast, cost-effective).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text or None if failed
        """
        try:
            doc = fitz.open(str(file_path))
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_content.append(f"--- PAGE {page_num + 1} ---\n{text}")
            
            doc.close()
            
            if text_content:
                return "\n\n".join(text_content)
            else:
                logger.warning(f"No text extracted from {file_path} using PyMuPDF")
                return None
                
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {file_path}: {e}")
            return None
    
    async def _extract_text_unstructured(self, file_path: Path) -> Optional[str]:
        """
        Extract text using Unstructured.io (handles complex layouts).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted text or None if failed
        """
        try:
            # For now, implement basic text extraction
            # In production, this would use the Unstructured.io library
            
            if file_path.suffix.lower() == '.docx':
                doc = DocxDocument(str(file_path))
                text_content = []
                
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_content.append(paragraph.text)
                
                return "\n".join(text_content)
            else:
                # Fallback to PyMuPDF for other formats
                return await self._extract_text_pymupdf(file_path)
                
        except Exception as e:
            logger.error(f"Unstructured extraction failed for {file_path}: {e}")
            return None
    
    async def _extract_text_document_ai(self, file_path: Path) -> Optional[str]:
        """
        Extract text using Google Document AI (high-fidelity).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted text or None if failed
        """
        try:
            # For now, fallback to PyMuPDF
            # In production, this would integrate with Google Document AI
            logger.info(f"Document AI extraction not yet implemented, using PyMuPDF fallback for {file_path}")
            return await self._extract_text_pymupdf(file_path)
            
        except Exception as e:
            logger.error(f"Document AI extraction failed for {file_path}: {e}")
            return None
    
    async def _intelligent_chunking(self, text_content: str, document: Document) -> List[Dict[str, Any]]:
        """
        Perform intelligent chunking preserving RFP structure.
        
        Args:
            text_content: Raw extracted text
            document: Document model instance
            
        Returns:
            List[Dict]: List of chunk data dictionaries
        """
        chunks = []
        
        # Split by pages first
        pages = text_content.split("--- PAGE ")
        
        for page_text in pages:
            if not page_text.strip():
                continue
            
            # Extract page number
            page_match = re.match(r'^(\d+) ---', page_text)
            page_num = int(page_match.group(1)) if page_match else 1
            
            # Remove page header
            page_content = re.sub(r'^\d+ ---\n', '', page_text)
            
            # Split by sections (look for RFP section patterns)
            sections = self._split_by_sections(page_content)
            
            for section_data in sections:
                # Split section into paragraphs
                paragraphs = self._split_by_paragraphs(section_data['text'])
                
                for para_idx, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                        continue
                    
                    chunk_data = {
                        'text': paragraph,
                        'cleaned_text': self._clean_text(paragraph),
                        'page': page_num,
                        'paragraph': para_idx + 1,
                        'section': section_data.get('section'),
                        'subsection': section_data.get('subsection'),
                        'type': 'paragraph',
                        'confidence': 'high'
                    }
                    chunks.append(chunk_data)
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text by RFP sections (Section A, B, C, etc.).
        
        Args:
            text: Text to split
            
        Returns:
            List[Dict]: List of section data
        """
        sections = []
        
        # Look for section patterns
        section_patterns = [
            r'Section\s+([A-Z])\s*[-:]?\s*(.*?)(?=Section\s+[A-Z]|$)',
            r'([A-Z])\s*[-:]?\s*(.*?)(?=[A-Z]\s*[-:]|$)',
        ]
        
        current_section = None
        current_text = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            section_match = None
            for pattern in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    section_match = match
                    break
            
            if section_match:
                # Save previous section
                if current_section and current_text:
                    sections.append({
                        'section': current_section,
                        'text': '\n'.join(current_text)
                    })
                
                # Start new section
                current_section = section_match.group(1)
                current_text = [line]
            else:
                current_text.append(line)
        
        # Add final section
        if current_section and current_text:
            sections.append({
                'section': current_section,
                'text': '\n'.join(current_text)
            })
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections.append({
                'section': 'UNKNOWN',
                'text': text
            })
        
        return sections
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of paragraphs
        """
        # Split by double newlines or bullet points
        paragraphs = re.split(r'\n\s*\n|\n\s*[•\-\*]\s*', text)
        
        # Clean up paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # Only keep substantial paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common document artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed text
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical text
        
        return text.strip()
    
    async def _extract_requirements(self, chunks: List[TextChunk], document: Document) -> List[Dict[str, Any]]:
        """
        Extract and classify requirements from text chunks.
        
        Args:
            chunks: List of text chunks
            document: Document model instance
            
        Returns:
            List[Dict]: List of requirement data
        """
        requirements = []
        
        for chunk in chunks:
            # Check if chunk contains requirement-like text
            if self._is_requirement_text(chunk.raw_text):
                classification = await self._classify_requirement(chunk.raw_text)
                
                requirement_data = {
                    'chunk_id': chunk.id,
                    'raw_text': chunk.raw_text,
                    'clean_text': chunk.cleaned_text,
                    'classification': classification,
                    'source_page': chunk.source_page,
                    'source_paragraph': chunk.source_paragraph,
                    'source_section': chunk.section_identifier,
                    'source_subsection': chunk.subsection_identifier,
                    'confidence': 'medium'
                }
                requirements.append(requirement_data)
        
        return requirements
    
    def _is_requirement_text(self, text: str) -> bool:
        """
        Determine if text contains a requirement.
        
        Args:
            text: Text to analyze
            
        Returns:
            bool: True if text appears to be a requirement
        """
        # Look for requirement indicators
        requirement_indicators = [
            r'shall\s+',
            r'must\s+',
            r'required\s+to\s+',
            r'contractor\s+shall\s+',
            r'vendor\s+shall\s+',
            r'system\s+shall\s+',
            r'application\s+shall\s+',
            r'provide\s+',
            r'deliver\s+',
            r'submit\s+',
            r'include\s+',
            r'ensure\s+',
            r'comply\s+with\s+',
            r'meet\s+the\s+following',
            r'minimum\s+requirements',
            r'technical\s+requirements',
            r'performance\s+requirements'
        ]
        
        text_lower = text.lower()
        
        # Check for requirement indicators
        for pattern in requirement_indicators:
            if re.search(pattern, text_lower):
                return True
        
        # Check for numbered requirements
        if re.search(r'^\d+\.\d+', text.strip()):
            return True
        
        # Check for bullet points that might be requirements
        if re.search(r'^[•\-\*]\s+', text.strip()):
            return True
        
        return False
    
    async def _classify_requirement(self, text: str) -> RequirementClassification:
        """
        Classify a requirement using AI.
        
        Args:
            text: Requirement text
            
        Returns:
            RequirementClassification: Classified requirement type
        """
        if not self.openai_available:
            # Fallback to rule-based classification
            return self._classify_requirement_rules(text)
        
        try:
            # Use OpenAI for classification
            prompt = f"""
            Classify the following RFP requirement text into one of these categories:
            - PERFORMANCE_REQUIREMENT: System performance, speed, capacity requirements
            - DELIVERABLE_REQUIREMENT: Specific deliverables, documents, or outputs
            - COMPLIANCE_REQUIREMENT: Regulatory, security, or compliance requirements
            - EVALUATION_CRITERIA: How proposals will be evaluated or scored
            - INSTRUCTION: Instructions for proposal submission
            - FAR_CLAUSE: Federal Acquisition Regulation clauses
            - OTHER: Anything else

            Text: "{text}"

            Respond with only the category name.
            """
            
            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            classification_text = response.choices[0].message.content.strip()
            
            # Map to enum
            classification_map = {
                'PERFORMANCE_REQUIREMENT': RequirementClassification.PERFORMANCE_REQUIREMENT,
                'DELIVERABLE_REQUIREMENT': RequirementClassification.DELIVERABLE_REQUIREMENT,
                'COMPLIANCE_REQUIREMENT': RequirementClassification.COMPLIANCE_REQUIREMENT,
                'EVALUATION_CRITERIA': RequirementClassification.EVALUATION_CRITERIA,
                'INSTRUCTION': RequirementClassification.INSTRUCTION,
                'FAR_CLAUSE': RequirementClassification.FAR_CLAUSE,
                'OTHER': RequirementClassification.OTHER
            }
            
            return classification_map.get(classification_text, RequirementClassification.OTHER)
            
        except Exception as e:
            logger.warning(f"AI classification failed: {e}, using rule-based fallback")
            return self._classify_requirement_rules(text)
    
    def _classify_requirement_rules(self, text: str) -> RequirementClassification:
        """
        Rule-based requirement classification fallback.
        
        Args:
            text: Requirement text
            
        Returns:
            RequirementClassification: Classified requirement type
        """
        text_lower = text.lower()
        
        # Performance requirements
        if any(word in text_lower for word in ['performance', 'speed', 'response time', 'capacity', 'throughput', 'latency']):
            return RequirementClassification.PERFORMANCE_REQUIREMENT
        
        # Deliverable requirements
        if any(word in text_lower for word in ['deliver', 'provide', 'submit', 'documentation', 'report', 'manual']):
            return RequirementClassification.DELIVERABLE_REQUIREMENT
        
        # Compliance requirements
        if any(word in text_lower for word in ['comply', 'security', 'regulatory', 'standard', 'certification', 'audit']):
            return RequirementClassification.COMPLIANCE_REQUIREMENT
        
        # Evaluation criteria
        if any(word in text_lower for word in ['evaluation', 'criteria', 'score', 'weight', 'points', 'assess']):
            return RequirementClassification.EVALUATION_CRITERIA
        
        # Instructions
        if any(word in text_lower for word in ['instruction', 'submit', 'format', 'deadline', 'proposal']):
            return RequirementClassification.INSTRUCTION
        
        # FAR clauses
        if any(word in text_lower for word in ['far', 'clause', 'federal acquisition']):
            return RequirementClassification.FAR_CLAUSE
        
        return RequirementClassification.OTHER


# Global processor instance
document_processor = DocumentProcessor()


async def process_document_async(document_id: uuid.UUID, job_id: uuid.UUID) -> bool:
    """
    Async wrapper for document processing.
    
    Args:
        document_id: UUID of the document to process
        job_id: UUID of the processing job
        
    Returns:
        bool: True if processing succeeded
    """
    return await document_processor.process_document(document_id, job_id)
