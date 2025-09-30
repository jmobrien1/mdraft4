#!/bin/bash

# RFP Extraction Platform Startup Script
# This script starts both the FastAPI backend and Streamlit frontend

echo "🚀 Starting RFP Extraction Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if database is initialized
if [ ! -f "rfp_extraction.db" ]; then
    echo "📊 Initializing database..."
    python database.py
fi

# Start FastAPI backend in background
echo "🔧 Starting FastAPI backend on http://localhost:8000"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend started successfully"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on http://localhost:8501"
echo "📋 Access the application at: http://localhost:8501"
echo "📚 API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

