#!/bin/bash

# Production startup script for Ollama Chat App
set -e

# Configuration
APP_NAME="ollama-chat-app"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/env-chat"
LOG_DIR="$APP_DIR/logs"
PID_FILE="$APP_DIR/app.pid"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_DIR/startup.log"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_DIR/startup.log"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_DIR/startup.log"
}

# Function to check if app is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# Function to stop the app
stop_app() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log "Stopping $APP_NAME (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local count=0
        while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            warn "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
        
        rm -f "$PID_FILE"
        log "App stopped"
    else
        log "App is not running"
    fi
}

# Function to start the app
start_app() {
    if is_running; then
        warn "App is already running"
        return 1
    fi
    
    log "Starting $APP_NAME in $ENVIRONMENT mode..."
    
    # Activate virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        error "Virtual environment not found at $VENV_DIR"
        error "Please run: python3 -m venv env-chat && source env-chat/bin/activate && pip install -r requirements.txt"
        return 1
    fi
    
    # Check if requirements are installed
    if ! source "$VENV_DIR/bin/activate" && python -c "import fastapi, uvicorn, ollama" 2>/dev/null; then
        error "Required packages not installed. Installing requirements..."
        source "$VENV_DIR/bin/activate"
        pip install -r requirements.txt
    fi
    
    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    export PYTHONPATH="$APP_DIR"
    
    # Start the app with gunicorn for production
    if [ "$ENVIRONMENT" = "production" ]; then
        log "Starting with Gunicorn (production mode)..."
        source "$VENV_DIR/bin/activate"
        nohup gunicorn \
            --bind "0.0.0.0:8000" \
            --workers 4 \
            --worker-class uvicorn.workers.UvicornWorker \
            --worker-connections 1000 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --timeout 30 \
            --keep-alive 2 \
            --preload \
            --pid "$PID_FILE" \
            --access-logfile "$LOG_DIR/access.log" \
            --error-logfile "$LOG_DIR/error.log" \
            --log-level info \
            app:app > "$LOG_DIR/app.log" 2>&1 &
    else
        log "Starting with Uvicorn (development mode)..."
        source "$VENV_DIR/bin/activate"
        nohup uvicorn \
            app:app \
            --host 0.0.0.0 \
            --port 8000 \
            --reload \
            --log-level info \
            --access-log \
            --log-config "$APP_DIR/logging.conf" > "$LOG_DIR/app.log" 2>&1 &
        echo $! > "$PID_FILE"
    fi
    
    # Wait for app to start
    local count=0
    while [ $count -lt 30 ]; do
        if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
            log "App started successfully!"
            log "Health check: $(curl -s http://localhost:8000/api/health | jq -r '.status' 2>/dev/null || echo 'unknown')"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    error "App failed to start within 30 seconds"
    return 1
}

# Function to show status
show_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log "App is running (PID: $pid)"
        
        # Show health status
        if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
            local health=$(curl -s http://localhost:8000/api/health)
            echo "Health Status:"
            echo "$health" | jq '.' 2>/dev/null || echo "$health"
        else
            warn "Health check failed"
        fi
        
        # Show process info
        echo "Process Info:"
        ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || echo "Process not found"
    else
        log "App is not running"
    fi
}

# Function to show logs
show_logs() {
    local log_file="$1"
    if [ -z "$log_file" ]; then
        log_file="$LOG_DIR/app.log"
    fi
    
    if [ -f "$log_file" ]; then
        echo "=== $log_file ==="
        tail -n 50 "$log_file"
    else
        error "Log file not found: $log_file"
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        stop_app
        sleep 2
        start_app
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [log_file]}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the application"
        echo "  stop    - Stop the application"
        echo "  restart - Restart the application"
        echo "  status  - Show application status"
        echo "  logs    - Show recent logs (default: app.log)"
        echo ""
        echo "Environment variables:"
        echo "  ENVIRONMENT - Set to 'production', 'staging', or 'development'"
        echo ""
        exit 1
        ;;
esac
