#!/usr/bin/env bash
set -e

NUM_WORKERS=""

while getopts "n:" opt; do
    case $opt in
        n) NUM_WORKERS="$OPTARG" ;;
        *) echo "Usage: $0 -n <num_workers>"; exit 1 ;;
    esac
done

if [ -z "$NUM_WORKERS" ]; then
    echo "Error: -n <num_workers> is required"
    echo "Usage: $0 -n <num_workers>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for i in $(seq 1 "$NUM_WORKERS"); do
    PORT=$((8000 + i))
    RCFILE=$(mktemp /tmp/splitbox_worker_XXXXXX.sh)
    cat > "$RCFILE" << EOF
source '$SCRIPT_DIR/init_env.sh'
cd '$SCRIPT_DIR/SplitBox'
history -s 'python worker.py -mode local -p $PORT'
python worker.py -mode local -p $PORT
EOF
    echo "Starting worker $i on port $PORT"
    gnome-terminal --title="Worker $i (port $PORT)" -- bash --rcfile "$RCFILE"
done

echo "Started $NUM_WORKERS worker(s) on ports 8001-$((8000 + NUM_WORKERS))"
