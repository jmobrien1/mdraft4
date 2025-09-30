#!/usr/bin/env python3
"""
Test script for RFP Extraction Platform

This script tests the core functionality of the system to ensure
everything is working correctly.
"""

import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_FILE_PATH = "test_document.txt"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "healthy":
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {data}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_system_stats():
    """Test the system statistics endpoint"""
    print("📊 Testing system statistics...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ System stats retrieved:")
        print(f"   - Total documents: {data['total_documents']}")
        print(f"   - Total requirements: {data['total_requirements']}")
        print(f"   - Total chunks: {data['total_chunks']}")
        return True
    except Exception as e:
        print(f"❌ System stats error: {e}")
        return False

def test_document_list():
    """Test the document list endpoint"""
    print("📁 Testing document list...")
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Document list retrieved: {len(data)} documents")
        return True
    except Exception as e:
        print(f"❌ Document list error: {e}")
        return False

def test_review_queue():
    """Test the review queue endpoint"""
    print("📝 Testing review queue...")
    try:
        response = requests.get(f"{API_BASE_URL}/requirements/review_queue")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Review queue retrieved: {data['total_count']} items pending")
        return True
    except Exception as e:
        print(f"❌ Review queue error: {e}")
        return False

def test_search():
    """Test the search endpoint"""
    print("🔍 Testing search functionality...")
    try:
        search_data = {
            "query": "requirement",
            "limit": 10
        }
        
        response = requests.post(f"{API_BASE_URL}/search", json=search_data)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Search completed: {data['total_count']} results found")
        return True
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def create_test_document():
    """Create a test document for upload testing"""
    test_content = """
    RFP Test Document
    
    Section C - Technical Requirements
    
    3.1 Performance Requirements
    The contractor shall provide a system that meets the following performance criteria:
    - Response time shall not exceed 2 seconds
    - System availability shall be 99.9%
    - Data processing capacity shall handle 10,000 transactions per hour
    
    3.2 Deliverable Requirements
    The contractor shall deliver:
    - System documentation
    - User training materials
    - Source code and binaries
    
    Section L - Proposal Instructions
    
    L.1 Proposal Format
    Proposals shall be submitted in the following format:
    - Executive Summary (2 pages maximum)
    - Technical Approach (10 pages maximum)
    - Cost Proposal (separate volume)
    
    Section M - Evaluation Criteria
    
    M.1 Evaluation Factors
    Proposals will be evaluated based on:
    - Technical Approach (40 points)
    - Past Performance (30 points)
    - Cost (30 points)
    """
    
    with open(TEST_FILE_PATH, "w") as f:
        f.write(test_content)
    
    print(f"📄 Test document created: {TEST_FILE_PATH}")

def test_document_upload():
    """Test document upload functionality"""
    print("📤 Testing document upload...")
    
    # Create test document
    create_test_document()
    
    try:
        with open(TEST_FILE_PATH, "rb") as f:
            files = {"file": (TEST_FILE_PATH, f, "text/plain")}
            response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ Document uploaded successfully:")
            print(f"   - Job ID: {data['job_id']}")
            print(f"   - Document ID: {data['document_id']}")
            return data['document_id']
    except Exception as e:
        print(f"❌ Document upload error: {e}")
        return None

def test_document_status(document_id):
    """Test document status checking"""
    if not document_id:
        return False
        
    print("📊 Testing document status...")
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{document_id}/status")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Document status retrieved: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Document status error: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    if Path(TEST_FILE_PATH).exists():
        Path(TEST_FILE_PATH).unlink()
        print(f"🧹 Cleaned up test file: {TEST_FILE_PATH}")

def main():
    """Run all tests"""
    print("🧪 RFP Extraction Platform - System Test")
    print("=" * 50)
    
    # Wait for backend to be ready
    print("⏳ Waiting for backend to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
        print(f"   Attempt {i+1}/{max_retries}...")
    else:
        print("❌ Backend not ready after 30 seconds")
        return False
    
    print("✅ Backend is ready!")
    print()
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("System Statistics", test_system_stats),
        ("Document List", test_document_list),
        ("Review Queue", test_review_queue),
        ("Search Functionality", test_search),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    # Test document upload (optional)
    print(f"\n🔬 Running Document Upload Test...")
    document_id = test_document_upload()
    if document_id:
        passed += 1
        total += 1
        test_document_status(document_id)
    
    # Cleanup
    cleanup_test_files()
    
    # Results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

