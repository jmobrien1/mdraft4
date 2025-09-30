#!/bin/bash

# RFP Extraction Platform - Service Starter
# This script starts both FastAPI backend and Streamlit frontend

echo "🚀 Starting RFP Extraction Platform Services..."

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

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    # Kill background processes
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start FastAPI backend in background
echo "🔧 Starting FastAPI backend on http://localhost:8000"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend started successfully"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start Streamlit frontend in background
echo "🎨 Starting Streamlit frontend on http://localhost:8501"
python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to start..."
sleep 10

# Check if Streamlit is running
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Streamlit started successfully"
else
    echo "❌ Streamlit failed to start"
    kill $BACKEND_PID $STREAMLIT_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Both services are running!"
echo "📋 Streamlit App: http://localhost:8501"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
wait

