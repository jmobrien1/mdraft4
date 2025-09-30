#!/usr/bin/env python3
"""
Debug script for document processing
"""

import asyncio
import uuid
from document_processor import document_processor
from database import get_db_session
from models import Document

async def debug_processing():
    """Debug document processing"""
    
    # Get the latest document
    with get_db_session() as db:
        document = db.query(Document).filter(
            Document.original_filename == "test_rfp.txt"
        ).first()
        
        if not document:
            print("No test document found")
            return
        
        print(f"Found document: {document.original_filename}")
        print(f"Status: {document.status}")
        print(f"File path: {document.file_path}")
        
        # Test text extraction directly
        from pathlib import Path
        file_path = Path(document.file_path)
        
        if file_path.exists():
            print(f"File exists: {file_path}")
            
            # Test PyMuPDF extraction
            try:
                text = await document_processor._extract_text_pymupdf(file_path)
                if text:
                    print(f"Text extracted successfully: {len(text)} characters")
                    print(f"First 200 chars: {text[:200]}")
                    
                    # Test chunking
                    chunks = await document_processor._intelligent_chunking(text, document)
                    print(f"Created {len(chunks)} chunks")
                    
                    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                        print(f"Chunk {i+1}: {chunk['text'][:100]}...")
                        
                        # Test requirement detection
                        is_req = document_processor._is_requirement_text(chunk['text'])
                        print(f"  Is requirement: {is_req}")
                        
                        if is_req:
                            classification = await document_processor._classify_requirement(chunk['text'])
                            print(f"  Classification: {classification}")
                else:
                    print("No text extracted")
            except Exception as e:
                print(f"Error during processing: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File does not exist: {file_path}")

if __name__ == "__main__":
    asyncio.run(debug_processing())

