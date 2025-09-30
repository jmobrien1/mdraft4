#!/usr/bin/env python3
"""
Manually trigger document processing for testing
"""

import asyncio
import uuid
from document_processor import process_document_async
from database import get_db_session
from models import Document, ProcessingJob

async def trigger_processing():
    """Manually trigger processing for the test document"""
    
    with get_db_session() as db:
        # Get the test document
        document = db.query(Document).filter(
            Document.original_filename == "test_rfp.txt"
        ).first()
        
        if not document:
            print("No test document found")
            return
        
        print(f"Processing document: {document.original_filename}")
        
        # Create a processing job
        job = ProcessingJob(
            id=uuid.uuid4(),
            document_id=document.id,
            job_type="extraction",
            status="pending"
        )
        db.add(job)
        db.commit()
        
        print(f"Created job: {job.id}")
        
        # Process the document
        success = await process_document_async(document.id, job.id)
        
        print(f"Processing completed: {success}")
        
        # Check results
        if success:
            from models import TextChunk, Requirement
            chunks = db.query(TextChunk).filter(TextChunk.document_id == document.id).all()
            requirements = db.query(Requirement).filter(Requirement.document_id == document.id).all()
            
            print(f"Created {len(chunks)} text chunks")
            print(f"Created {len(requirements)} requirements")
            
            for i, req in enumerate(requirements[:5]):  # Show first 5 requirements
                print(f"Requirement {i+1}: {req.raw_text[:100]}...")
                print(f"  Classification: {req.classification}")
                print(f"  Confidence: {req.ai_confidence_score}")

if __name__ == "__main__":
    asyncio.run(trigger_processing())

