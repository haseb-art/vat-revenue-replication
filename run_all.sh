#!/usr/bin/env bash
# Run the full replication from the repository root.
set -euo pipefail
cd "$(dirname "$0")/code"
python3 00_load_data.py
python3 01_descriptives.py > ../out/01_descriptives.txt
python3 02_twfe.py > ../out/02_twfe.txt
python3 03_callaway_santanna.py > ../out/03_callaway_santanna.txt
python3 04_figures.py
echo "Done. Outputs in out/."
