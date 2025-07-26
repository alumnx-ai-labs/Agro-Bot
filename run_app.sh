#!/bin/bash
# run_setup.sh - Simple script to run both frontend and backend

echo "🌱 Farmer Assistant MVP - Quick Setup"
echo "====================================="

# Check if we're in the right directory and find main.py
MAIN_PY_PATH=""
if [ -f "main.py" ]; then
    MAIN_PY_PATH="main.py"
elif [ -f "cloud_functions/main.py" ]; then
    MAIN_PY_PATH="cloud_functions/main.py"
else
    echo "❌ Error: main.py not found!"
    exit 1
fi

if [ ! -f "webapp/app.py" ]; then
    echo "❌ Error: webapp/app.py not found!"
    exit 1
fi

echo "✅ Found main.py at: $MAIN_PY_PATH"
echo "✅ Found webapp/app.py"

# Set environment
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export CLOUD_FUNCTION_URL="http://localhost:8080"

# Create logs directory
mkdir -p logs

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    jobs -p | xargs kill 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    lsof -ti:5500 | xargs kill -9 2>/dev/null || true
    echo "✅ Stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Kill any existing processes on our ports
echo "🔄 Freeing ports..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:5500 | xargs kill -9 2>/dev/null || true
sleep 1

echo ""
echo "🚀 Starting services..."

# Start Backend
echo "📡 Starting backend (port 8080)..."
functions-framework \
    --target=farmer_assistant \
    --source="$MAIN_PY_PATH" \
    --port=8080 \
    --debug \
    --host=0.0.0.0 > logs/backend.log 2>&1 &

BACKEND_PID=$!

# Wait for backend
echo "⏳ Waiting for backend..."
for i in {1..20}; do
    if curl -s http://localhost:8080/health_check >/dev/null 2>&1; then
        echo "✅ Backend ready!"
        break
    fi
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend failed to start!"
        echo "📜 Backend log:"
        tail -10 logs/backend.log
        exit 1
    fi
    sleep 1
done

# Start Frontend
echo "🌐 Starting frontend (port 5500)..."
(cd webapp && python app.py) > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend
echo "⏳ Waiting for frontend..."
for i in {1..15}; do
    if curl -s http://localhost:5500/health >/dev/null 2>&1; then
        echo "✅ Frontend ready!"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend failed to start!"
        echo "📜 Frontend log:"
        tail -10 logs/frontend.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "🎉 SUCCESS! Services are running:"
echo "================================="
echo "📱 Frontend: http://localhost:5500"
echo "🔧 Backend:  http://localhost:8080"
echo ""
echo "📋 View logs:"
echo "   Backend:  tail -f logs/backend.log"
echo "   Frontend: tail -f logs/frontend.log"
echo ""
echo "🛑 Press Ctrl+C to stop both services"
echo ""

# Keep running until user stops
while true; do
    # Check if both services are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend stopped!"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend stopped!"
        break
    fi
    
    sleep 5
done

cleanup