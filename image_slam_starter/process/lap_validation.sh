#!/bin/bash
# Lap-wise validation: for each configuration, hold out all loop pairs of one lap (2, 1, 0) and report how far
# that lap (held only by odometry) ends up from its loop pairs: translation rmse [m] / rotation median [deg].
# Runs one job at a time (the triangulation stage needs the 8 GB GPU for itself).
# Usage: process/lap_validation.sh            (about 3 min per run with --height-scale, 20 s without)
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
run() {
  name=$1; shift
  line="$name"
  for lap in 2 1 0; do
    out=$($PY -u slam_v2.py --holdout-lap $lap --out _val.csv "$@" 2>&1 | grep "loops (held)" | head -1)
    line="$line | lap$lap: $(echo "$out" | sed 's/.*rmse \([0-9.]*\) m.*rot median \([0-9.]*\) deg/\1 \/ \2/')"
  done
  echo "$line"
}
run "final (height-scale 1)" 
run "height-scale 0        " --height-scale 0
run "height-scale 2        " --height-scale 2
run "no car constraint     " --no-nhc --height-scale 0
run "no gyro               " --no-gyro --height-scale 0
run "zmax 15               " --zmax 15 --height-scale 0
run "ground only scale     " --no-scale-graph --height-scale 0
