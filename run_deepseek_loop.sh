#!/bin/bash
# Run DeepSeek manager in continuous loop with logging

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file with timestamp
LOG_FILE="logs/deepseek_manager_$(date +%Y%m%d_%H%M%S).log"

echo "Starting DeepSeek manager loop..."
echo "Logging to: $LOG_FILE"
echo "Press Ctrl+C to stop"

# Run in loop with 15-minute poll interval
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "=== $TIMESTAMP ===" >> "$LOG_FILE"
    python scripts/run_deepseek_manager.py --loop --poll-sec 900 2>&1 | tee -a "$LOG_FILE"
    
    # If the script exits (e.g., error), wait and restart
    echo "Manager exited, restarting in 60 seconds..." >> "$LOG_FILE"
    sleep 60
done