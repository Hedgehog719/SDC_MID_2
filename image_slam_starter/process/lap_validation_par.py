"""Parallel lap-wise validation: every (config, lap) is one job, N at a time (~3.1 GB GPU, ~8 GB RAM, ~18 cores per run).
usage: python process/lap_validation_par.py N [--stagger S] "name|args" ["name|args" ...]   -> one line per config (same format as val_vo.sh) + mean"""
import concurrent.futures as cf, os, re, shlex, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "Track1", "result", "val_logs")      # per-run logs
subprocess.run(["mkdir", "-p", D])
N = int(sys.argv[1])
argv = sys.argv[2:]
stagger = 0.0          # --stagger S: start the first N jobs S seconds apart (their GPU peaks do not coincide)
if argv and argv[0] == "--stagger":
    stagger, argv = float(argv[1]), argv[2:]
cfgs = [c.split("|", 1) for c in argv]
jobs = [(j, name, lap, a) for j, (name, lap, a) in enumerate((name, lap, a) for name, a in cfgs for lap in (2, 1, 0))]

def run(job):
    j, name, lap, a = job
    if j < N:
        time.sleep(j * stagger)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    log = f"{D}/{tag}_{lap}.log"
    cmd = ["../.venv/bin/python", "-u", "slam_v2.py", "--holdout-lap", str(lap), "--out", f"_vp_{tag}_{lap}.csv"] + shlex.split(a)
    for attempt in range(4):          # GPU peaks of parallel runs can coincide: retry a run that ran out of GPU memory
        with open(log, "w") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        txt = open(log).read()
        if not any(e in txt for e in ("OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED", "CUDA out of memory")):
            break
        print(f"  {name} lap{lap}: out of GPU memory, retry {attempt + 1} in 60 s", flush=True)
        time.sleep(60)
    m = re.search(r"loops \(held\):.*rmse ([0-9.]+) m.*rot median ([0-9.]+) deg", open(log).read())
    r = (float(m.group(1)), float(m.group(2))) if m else None
    print(f"  done {name} lap{lap}: {r}  ({time.strftime('%H:%M:%S')})", flush=True)
    return r

with cf.ThreadPoolExecutor(N) as ex:
    res = list(ex.map(run, jobs))
for i, (name, a) in enumerate(cfgs):
    rr = res[3 * i:3 * i + 3]
    line = " | ".join(f"lap{lap}: " + (f"{r[0]:.3f} / {r[1]:.2f}" if r else "FAIL") for lap, r in zip((2, 1, 0), rr))
    ok = [r[0] for r in rr if r]
    print(f"{name} | {line} | mean {sum(ok)/len(ok):.3f}" if len(ok) == 3 else f"{name} | {line}", flush=True)
