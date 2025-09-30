"""
Pydantic schemas for API request/response validation.

This module defines the data validation schemas used by the FastAPI backend
for request validation, response serialization, and data transformation.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum

from models import (
    ProcessingStatus, RequirementClassification, ValidationStatus,
    create_audit_record
)


# Base schemas for common fields
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AuditMixin(BaseModel):
    """Mixin for audit trail fields"""
    history: List[Dict[str, Any]] = Field(default_factory=list)


# Document schemas
class DocumentBase(BaseModel):
    """Base document schema"""
    filename: str = Field(..., max_length=255, description="Original filename")
    file_type: str = Field(..., max_length=50, description="File type (PDF, DOCX, etc.)")
    file_size: int = Field(..., gt=0, description="File size in bytes")


class DocumentCreate(DocumentBase):
    """Schema for document creation"""
    pass


class DocumentUpdate(BaseModel):
    """Schema for document updates"""
    status: Optional[ProcessingStatus] = None
    processing_engine_used: Optional[str] = None
    extraction_confidence: Optional[str] = None


class DocumentResponse(DocumentBase):
    """Schema for document responses"""
    id: UUID
    original_filename: str
    file_path: str
    mime_type: str
    status: ProcessingStatus
    processing_engine_used: Optional[str] = None
    extraction_confidence: Optional[str] = None
    uploaded_by: str
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Text chunk schemas
class TextChunkBase(BaseModel):
    """Base text chunk schema"""
    chunk_index: int = Field(..., ge=0)
    section_identifier: Optional[str] = Field(None, max_length=100)
    subsection_identifier: Optional[str] = Field(None, max_length=100)
    raw_text: str
    cleaned_text: Optional[str] = None
    source_page: Optional[int] = Field(None, ge=1)
    source_paragraph: Optional[int] = Field(None, ge=1)
    chunk_type: Optional[str] = Field(None, max_length=50)
    confidence_score: Optional[str] = Field(None, max_length=20)


class TextChunkCreate(TextChunkBase):
    """Schema for text chunk creation"""
    document_id: UUID


class TextChunkResponse(TextChunkBase, TimestampMixin):
    """Schema for text chunk responses"""
    id: UUID
    document_id: UUID
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None
    
    class Config:
        from_attributes = True


# Requirement schemas
class RequirementBase(BaseModel):
    """Base requirement schema"""
    raw_text: str
    clean_text: Optional[str] = None
    classification: RequirementClassification = RequirementClassification.OTHER
    source_page: int = Field(..., ge=1)
    source_paragraph: Optional[int] = Field(None, ge=1)
    source_section: Optional[str] = Field(None, max_length=100)
    source_subsection: Optional[str] = Field(None, max_length=100)
    ai_confidence_score: Optional[str] = Field(None, max_length=20)
    validation_notes: Optional[str] = None


class RequirementCreate(RequirementBase):
    """Schema for requirement creation"""
    document_id: UUID
    source_chunk_id: UUID
    extraction_method: Optional[str] = Field(None, max_length=50)


class RequirementUpdate(BaseModel):
    """Schema for requirement updates"""
    clean_text: Optional[str] = None
    classification: Optional[RequirementClassification] = None
    status: Optional[ValidationStatus] = None
    validation_notes: Optional[str] = None


class RequirementResponse(RequirementBase, TimestampMixin, AuditMixin):
    """Schema for requirement responses"""
    id: UUID
    document_id: UUID
    source_chunk_id: UUID
    extraction_method: Optional[str] = None
    status: ValidationStatus
    validated_by: Optional[str] = None
    validated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Cross-reference schemas
class CrossReferenceBase(BaseModel):
    """Base cross-reference schema"""
    reference_type: str = Field(..., max_length=50)
    reference_text: str
    reference_target: Optional[str] = Field(None, max_length=200)
    confidence_score: Optional[str] = Field(None, max_length=20)
    similarity_score: Optional[str] = Field(None, max_length=20)


class CrossReferenceCreate(CrossReferenceBase):
    """Schema for cross-reference creation"""
    requirement_id: UUID
    target_chunk_id: UUID


class CrossReferenceResponse(CrossReferenceBase, TimestampMixin):
    """Schema for cross-reference responses"""
    id: UUID
    requirement_id: UUID
    target_chunk_id: UUID
    created_by: str
    
    class Config:
        from_attributes = True


# Processing job schemas
class ProcessingJobBase(BaseModel):
    """Base processing job schema"""
    job_type: str = Field(..., max_length=50)
    status: str = Field(..., max_length=50)
    total_items: int = Field(default=0, ge=0)
    processed_items: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)


class ProcessingJobCreate(ProcessingJobBase):
    """Schema for processing job creation"""
    document_id: UUID


class ProcessingJobResponse(ProcessingJobBase, TimestampMixin):
    """Schema for processing job responses"""
    id: UUID
    document_id: UUID
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# File upload schemas
class FileUploadResponse(BaseModel):
    """Schema for file upload responses"""
    job_id: UUID
    document_id: UUID
    message: str
    status: str


class FileUploadStatus(BaseModel):
    """Schema for file upload status"""
    job_id: Optional[UUID] = None
    status: str
    progress: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Review queue schemas
class ReviewQueueItem(BaseModel):
    """Schema for items in the review queue"""
    id: UUID
    document_id: UUID
    raw_text: str
    clean_text: Optional[str] = None
    classification: RequirementClassification
    ai_confidence_score: Optional[str] = None
    source_page: int
    source_section: Optional[str] = None
    source_subsection: Optional[str] = None
    status: ValidationStatus
    created_at: datetime
    
    class Config:
        from_attributes = True


class ReviewQueueResponse(BaseModel):
    """Schema for review queue responses"""
    items: List[ReviewQueueItem]
    total_count: int
    page: int
    page_size: int
    has_more: bool


class ValidationRequest(BaseModel):
    """Schema for validation requests"""
    clean_text: Optional[str] = None
    classification: Optional[RequirementClassification] = None
    validation_notes: Optional[str] = None
    action: str = Field(..., description="Action: 'approve' or 'correct'")


# Search and filter schemas
class SearchRequest(BaseModel):
    """Schema for search requests"""
    query: str = Field(..., min_length=1, max_length=500)
    document_ids: Optional[List[UUID]] = None
    classifications: Optional[List[RequirementClassification]] = None
    statuses: Optional[List[ValidationStatus]] = None
    limit: int = Field(default=50, ge=1, le=200)


class SearchResult(BaseModel):
    """Schema for search results"""
    id: UUID
    document_id: UUID
    raw_text: str
    clean_text: Optional[str] = None
    classification: RequirementClassification
    source_page: int
    source_section: Optional[str] = None
    similarity_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Schema for search responses"""
    results: List[SearchResult]
    total_count: int
    query: str


# Compliance matrix schemas
class ComplianceMatrixItem(BaseModel):
    """Schema for compliance matrix items"""
    requirement_id: UUID
    requirement_text: str
    classification: RequirementClassification
    source_section: str
    source_subsection: Optional[str] = None
    source_page: int
    related_instructions: List[str] = Field(default_factory=list)
    evaluation_criteria: List[str] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)
    status: ValidationStatus
    
    class Config:
        from_attributes = True


class ComplianceMatrixResponse(BaseModel):
    """Schema for compliance matrix responses"""
    document_id: UUID
    document_name: str
    items: List[ComplianceMatrixItem]
    total_requirements: int
    validated_requirements: int
    pending_requirements: int
    generated_at: datetime


# Statistics and analytics schemas
class DocumentStats(BaseModel):
    """Schema for document statistics"""
    document_id: UUID
    total_chunks: int
    total_requirements: int
    requirements_by_classification: Dict[str, int]
    validation_status_counts: Dict[str, int]
    processing_time_seconds: Optional[float] = None
    confidence_distribution: Dict[str, int]


class SystemStats(BaseModel):
    """Schema for system statistics"""
    total_documents: int
    total_requirements: int
    total_chunks: int
    documents_by_status: Dict[str, int]
    requirements_by_classification: Dict[str, int]
    validation_status_counts: Dict[str, int]
    average_processing_time: Optional[float] = None
    system_health: Dict[str, Any]


# Error response schemas
class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationErrorResponse(BaseModel):
    """Schema for validation error responses"""
    error: str = "Validation Error"
    detail: str
    field_errors: Dict[str, List[str]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health check schemas
class HealthCheckResponse(BaseModel):
    """Schema for health check responses"""
    status: str
    database_connection: bool
    tables_exist: bool
    pgvector_available: bool
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Configuration schemas
class ProcessingConfig(BaseModel):
    """Schema for processing configuration"""
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    confidence_threshold: str = Field(default="medium", pattern="^(low|medium|high)$")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    supported_formats: List[str] = Field(default=["pdf", "docx"])


class SystemConfig(BaseModel):
    """Schema for system configuration"""
    processing_config: ProcessingConfig
    database_config: Dict[str, Any]
    ai_config: Dict[str, Any]
    storage_config: Dict[str, Any]


# Utility functions for schema validation
def validate_confidence_score(score: str) -> str:
    """Validate confidence score values"""
    valid_scores = ["low", "medium", "high"]
    if score not in valid_scores:
        raise ValueError(f"Confidence score must be one of: {valid_scores}")
    return score


def validate_file_type(file_type: str) -> str:
    """Validate file type values"""
    valid_types = ["pdf", "docx", "txt"]
    if file_type.lower() not in valid_types:
        raise ValueError(f"File type must be one of: {valid_types}")
    return file_type.lower()


# Custom validators
class RequirementCreateValidator:
    """Custom validator for requirement creation"""
    
    @staticmethod
    def validate_source_page(page: int) -> int:
        if page < 1:
            raise ValueError("Source page must be greater than 0")
        return page
    
    @staticmethod
    def validate_text_length(text: str) -> str:
        if len(text.strip()) < 10:
            raise ValueError("Requirement text must be at least 10 characters")
        if len(text) > 10000:
            raise ValueError("Requirement text must be less than 10,000 characters")
        return text.strip()


# Add validators to schemas
RequirementCreate.__validators__ = {
    'source_page': RequirementCreateValidator.validate_source_page,
    'raw_text': RequirementCreateValidator.validate_text_length
}
