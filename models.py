"""
Database models for RFP Extraction and Analysis Platform

This module defines the SQLAlchemy models that implement the core database schema
as specified in the RFP blueprint. The schema is designed for compliance and auditability
in government contracting, with full traceability of all data transformations.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, JSON, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import JSON
import uuid
from datetime import datetime
from enum import Enum
import os

# Conditional import for pgvector (PostgreSQL only)
try:
    from pgvector.sqlalchemy import Vector
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    # Create a placeholder Vector class for SQLite
    class Vector:
        def __init__(self, dimension):
            self.dimension = dimension

Base = declarative_base()


class ProcessingStatus(str, Enum):
    """Enumeration of document processing statuses"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTION_COMPLETE = "extraction_complete"
    PENDING_VALIDATION = "pending_validation"
    VALIDATION_COMPLETE = "validation_complete"
    ERROR = "error"


class RequirementClassification(str, Enum):
    """Enumeration of requirement classifications"""
    PERFORMANCE_REQUIREMENT = "PERFORMANCE_REQUIREMENT"
    DELIVERABLE_REQUIREMENT = "DELIVERABLE_REQUIREMENT"
    COMPLIANCE_REQUIREMENT = "COMPLIANCE_REQUIREMENT"
    EVALUATION_CRITERIA = "EVALUATION_CRITERIA"
    INSTRUCTION = "INSTRUCTION"
    FAR_CLAUSE = "FAR_CLAUSE"
    OTHER = "OTHER"


class ValidationStatus(str, Enum):
    """Enumeration of validation statuses for extracted data"""
    AI_EXTRACTED = "ai_extracted"
    HUMAN_VALIDATED = "human_validated"
    HUMAN_CORRECTED = "human_corrected"
    FLAGGED_FOR_REVIEW = "flagged_for_review"


class Document(Base):
    """
    Core document table storing metadata about uploaded RFP documents.
    
    This table serves as the primary reference for all documents in the system,
    maintaining the audit trail required for government contracting compliance.
    """
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)  # Path in GCS
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)  # PDF, DOCX, etc.
    mime_type = Column(String(100), nullable=False)
    
    # Processing metadata
    status = Column(String(50), nullable=False, default=ProcessingStatus.UPLOADED)
    processing_engine_used = Column(String(50))  # "pymupdf", "unstructured", "document_ai"
    extraction_confidence = Column(String(20))  # "high", "medium", "low"
    
    # Audit fields
    uploaded_by = Column(String(100), nullable=False)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processed_at = Column(DateTime)
    validated_at = Column(DateTime)
    
    # Relationships
    requirements = relationship("Requirement", back_populates="document", cascade="all, delete-orphan")
    text_chunks = relationship("TextChunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_documents_status', 'status'),
        Index('idx_documents_uploaded_at', 'uploaded_at'),
        Index('idx_documents_file_type', 'file_type'),
    )


class TextChunk(Base):
    """
    Text chunks extracted from documents with vector embeddings.
    
    This table stores the intelligently chunked text from RFP documents,
    preserving semantic boundaries and context for AI analysis.
    """
    __tablename__ = "text_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)
    section_identifier = Column(String(100))  # "Section C", "Section L", etc.
    subsection_identifier = Column(String(100))  # "3.1.1", "L.2.3", etc.
    
    # Text content
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text)
    
    # Source location tracking
    source_page = Column(Integer)
    source_paragraph = Column(Integer)
    source_line_start = Column(Integer)
    source_line_end = Column(Integer)
    
    # Vector embedding for semantic search
    embedding = Column(Vector(384) if VECTOR_AVAILABLE and not os.getenv("DATABASE_URL", "").startswith("sqlite") else Text)  # Sentence transformer dimension
    
    # Processing metadata
    chunk_type = Column(String(50))  # "paragraph", "table", "list", "header"
    confidence_score = Column(String(20))  # "high", "medium", "low"
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="text_chunks")
    requirements = relationship("Requirement", back_populates="source_chunk")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_text_chunks_document_id', 'document_id'),
        Index('idx_text_chunks_section', 'section_identifier'),
        # Only create vector index for PostgreSQL with pgvector
        *((Index('idx_text_chunks_embedding', 'embedding', postgresql_using='ivfflat'),) if VECTOR_AVAILABLE and not os.getenv("DATABASE_URL", "").startswith("sqlite") else ()),
    )


class Requirement(Base):
    """
    Extracted requirements and their classifications.
    
    This is the core table storing all extracted requirements with full
    audit trail for compliance and protest defense in government contracting.
    """
    __tablename__ = "requirements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    source_chunk_id = Column(UUID(as_uuid=True), ForeignKey("text_chunks.id"), nullable=False)
    
    # Requirement content
    raw_text = Column(Text, nullable=False)
    clean_text = Column(Text)
    classification = Column(String(50), nullable=False, default=RequirementClassification.OTHER)
    
    # Source location (redundant but critical for audit trail)
    source_page = Column(Integer, nullable=False)
    source_paragraph = Column(Integer)
    source_section = Column(String(100))  # "Section C", "Section L", etc.
    source_subsection = Column(String(100))  # "3.1.1", "L.2.3", etc.
    
    # AI processing metadata
    ai_confidence_score = Column(String(20))  # "high", "medium", "low"
    extraction_method = Column(String(50))  # "pymupdf", "unstructured", "document_ai"
    
    # Validation workflow
    status = Column(String(50), nullable=False, default=ValidationStatus.AI_EXTRACTED)
    validation_notes = Column(Text)
    
    # Audit trail - immutable history of all changes
    history = Column(JSON, default=list)  # Array of change records
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    validated_by = Column(String(100))
    validated_at = Column(DateTime)
    
    # Relationships
    document = relationship("Document", back_populates="requirements")
    source_chunk = relationship("TextChunk", back_populates="requirements")
    cross_references = relationship("CrossReference", back_populates="requirement", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_requirements_document_id', 'document_id'),
        Index('idx_requirements_status', 'status'),
        Index('idx_requirements_classification', 'classification'),
        Index('idx_requirements_source_section', 'source_section'),
        Index('idx_requirements_confidence', 'ai_confidence_score'),
    )


class CrossReference(Base):
    """
    Cross-references between requirements and related document sections.
    
    This table implements the knowledge graph functionality, linking
    related pieces of information scattered across RFP documents.
    """
    __tablename__ = "cross_references"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    requirement_id = Column(UUID(as_uuid=True), ForeignKey("requirements.id"), nullable=False)
    target_chunk_id = Column(UUID(as_uuid=True), ForeignKey("text_chunks.id"), nullable=False)
    
    # Reference metadata
    reference_type = Column(String(50), nullable=False)  # "attachment", "section", "clause", "instruction"
    reference_text = Column(Text, nullable=False)  # The actual reference text
    reference_target = Column(String(200))  # "Attachment 3", "Section M.2.1", etc.
    
    # AI confidence in the reference resolution
    confidence_score = Column(String(20))  # "high", "medium", "low"
    similarity_score = Column(String(20))  # Vector similarity score
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String(50), default="cross_reference_agent")
    
    # Relationships
    requirement = relationship("Requirement", back_populates="cross_references")
    target_chunk = relationship("TextChunk")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_cross_refs_requirement_id', 'requirement_id'),
        Index('idx_cross_refs_target_chunk_id', 'target_chunk_id'),
        Index('idx_cross_refs_type', 'reference_type'),
    )


class ProcessingJob(Base):
    """
    Background job tracking for document processing pipeline.
    
    This table tracks the status of asynchronous document processing jobs,
    enabling the frontend to poll for completion status.
    """
    __tablename__ = "processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Job metadata
    job_type = Column(String(50), nullable=False)  # "extraction", "validation", "cross_reference"
    status = Column(String(50), nullable=False, default="pending")  # "pending", "running", "completed", "failed"
    
    # Progress tracking
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSON)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    document = relationship("Document")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_processing_jobs_document_id', 'document_id'),
        Index('idx_processing_jobs_status', 'status'),
        Index('idx_processing_jobs_created_at', 'created_at'),
    )


class UserSession(Base):
    """
    User session tracking for audit and security.
    
    This table maintains session information for compliance and
    security auditing in the government contracting environment.
    """
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)
    session_token = Column(String(255), nullable=False, unique=True)
    
    # Session metadata
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_sessions_user_id', 'user_id'),
        Index('idx_user_sessions_token', 'session_token'),
        Index('idx_user_sessions_active', 'is_active'),
    )


# Database constraints and validations
def add_constraints():
    """Add database-level constraints for data integrity"""
    
    # Add check constraints for enum-like fields
    CheckConstraint(
        "status IN ('uploaded', 'processing', 'extraction_complete', 'pending_validation', 'validation_complete', 'error')",
        name='ck_documents_status'
    )
    
    CheckConstraint(
        "classification IN ('PERFORMANCE_REQUIREMENT', 'DELIVERABLE_REQUIREMENT', 'COMPLIANCE_REQUIREMENT', 'EVALUATION_CRITERIA', 'INSTRUCTION', 'FAR_CLAUSE', 'OTHER')",
        name='ck_requirements_classification'
    )
    
    CheckConstraint(
        "validation_status IN ('ai_extracted', 'human_validated', 'human_corrected', 'flagged_for_review')",
        name='ck_requirements_validation_status'
    )


# Utility functions for common operations
def create_audit_record(action: str, user: str, details: dict = None) -> dict:
    """
    Create a standardized audit record for the history field.
    
    Args:
        action: The action performed (e.g., 'ai_extraction', 'human_validation')
        user: The user or system that performed the action
        details: Additional details about the action
    
    Returns:
        Dictionary representing the audit record
    """
    return {
        "action": action,
        "user": user,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }


def add_audit_record(history: list, action: str, user: str, details: dict = None) -> list:
    """
    Add an audit record to the existing history list.
    
    Args:
        history: Existing history list
        action: The action performed
        user: The user or system that performed the action
        details: Additional details about the action
    
    Returns:
        Updated history list with new audit record
    """
    if history is None:
        history = []
    
    history.append(create_audit_record(action, user, details))
    return history
