#!/usr/bin/env python3
"""Roadmap Phase 4 gate: the scale ladder. BSIM4 (OSDI) inverter chains of growing size, one transient each,
engine against ngspice (same bsim4.osdi) for runtime and accuracy as long as ngspice finishes in --ng-timeout.

    python3 tests/scale_ladder.py [--stages 100,1000,10000,100000] [--ng-timeout 1800]

Accuracy: the 50 % crossing time of the chain's last driven stage that both simulators reach, and the
worst |dV| over all nodes, the engine's samples against ngspice's interpolated waveform.
Needs datasets/osdi/bsim4.osdi (tests/build_osdi.sh) and ngspice on PATH.
"""
import argparse, bisect, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE = os.path.join(ROOT, "build", "acutesim_engine_dump")

def chain(n, tstep, tstop):
    l = ["* scale ladder: %d-stage BSIM4 inverter chain" % n, ".osdi bsim4.osdi", "VDD 1 0 DC 1.2",
         "VIN 2 0 PULSE(0 1.2 0.1n 50p 50p 1n 2n)"]
    prev = "2"
    for i in range(n):
        l += [f"NP{i} o{i} {prev} 1 1 pch", f"NN{i} o{i} {prev} 0 0 nch"]
        prev = f"o{i}"
    l += [".model nch bsim4va type=1 w=1u l=100n", ".model pch bsim4va type=-1 w=2u l=100n",
          f".tran {tstep!r} {tstop!r}", ".end"]
    return "\n".join(l) + "\n"

def crossing(ts, vs, level=0.6):
    for i in range(1, len(ts)):
        a, b = vs[i - 1] - level, vs[i] - level
        if a * b < 0: return ts[i - 1] + (-a) * (ts[i] - ts[i - 1]) / (b - a)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="100,1000,10000,100000")
    ap.add_argument("--ng-timeout", type=float, default=1800)
    ap.add_argument("--tstop", type=float, default=1e-9)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="ladder_")
    shutil.copy(os.path.join(ROOT, "datasets", "osdi", "bsim4.osdi"), tmp)
    tstep = 10e-12
    print(f"{'stages':>8} {'devices':>9} {'engine s':>9} {'ngspice s':>10} {'speedup':>8} {'t50 last (ps) eng/ng':>22} {'worst dV':>9}")
    try:
        for n in map(int, a.stages.split(",")):
            cir = os.path.join(tmp, f"c{n}.cir")
            open(cir, "w").write(chain(n, tstep, a.tstop))
            # nodes reported: the probed ones, every stage up to the 40th (the edge travels ~25 ps/stage)
            probe = [f"o{i}" for i in range(min(n, 40))]
            t0 = time.time()
            out = subprocess.run([ENGINE, cir, "cpu", "tran", repr(tstep), repr(a.tstop)], capture_output=True,
                                 text=True, cwd=tmp).stdout
            te = time.time() - t0
            pts = [list(map(float, x.split())) for x in out.splitlines() if x[:1].isdigit()]
            if not pts or "CONVERGED 1" not in out:
                print(f"{n:>8} engine failed: {[x for x in out.splitlines() if x.startswith('ERROR')][:1]}"); continue
            # engine node order = first appearance: 1, 2, o0, o1, ...
            col = {f"o{i}": 3 + i for i in range(min(n, 40))}
            ng_time, t50e, t50n, worst = None, None, None, None
            last = None
            for i in reversed(range(min(n, 40))):
                if crossing([p[0] for p in pts], [p[col[f"o{i}"]] for p in pts]) is not None: last = f"o{i}"; break
            if last: t50e = crossing([p[0] for p in pts], [p[col[last]] for p in pts])
            ng = cir.replace(".cir", "_ng.cir")
            body = open(cir).read().replace(".osdi bsim4.osdi\n", "").replace(f".tran {tstep!r} {a.tstop!r}\n.end\n", "")
            dat = os.path.join(tmp, f"ng{n}.dat")
            open(ng, "w").write(body + ".control\npre_osdi bsim4.osdi\n"
                                f"tran {tstep!r} {a.tstop!r} 0 {tstep / 10!r}\nwrdata {dat} " +
                                " ".join(f"v({p})" for p in probe) + "\nquit\n.endc\n.end\n")
            ng_note = ""
            try:
                t0 = time.time()
                log = subprocess.run(["ngspice", "-b", ng], capture_output=True, text=True, cwd=tmp, timeout=a.ng_timeout)
                ng_time = time.time() - t0
                rows = [list(map(float, x.split())) for x in open(dat)]
                if len(rows) < 2:  # ngspice aborted (e.g. "Timestep too small" at 1e4 stages)
                    why = [x.strip() for x in (log.stdout + log.stderr).splitlines() if "too small" in x or "aborted" in x]
                    ng_note = "  ngspice failed: " + (why[0] if why else "no data")
                    raise ValueError(ng_note)
                xs = [r[0] for r in rows]
                ref = {p: [r[2 * k + 1] for r in rows] for k, p in enumerate(probe)}
                if last: t50n = crossing(xs, ref[last])
                worst = 0.0 if not ng_note else None
                for p in probe:
                    for row in pts:
                        t = row[0]
                        j = bisect.bisect_left(xs, t)
                        if j <= 0 or j >= len(xs): continue
                        w = (t - xs[j - 1]) / (xs[j] - xs[j - 1])
                        worst = max(worst, abs(row[col[p]] - (ref[p][j - 1] + w * (ref[p][j] - ref[p][j - 1]))))
            except (subprocess.TimeoutExpired, OSError, ValueError):
                ng_time = None if not ng_note else ng_time
            fmt = lambda x: "-" if x is None else f"{x * 1e12:.2f}"
            print(f"{n:>8} {2 * n:>9} {te:>9.2f} {('>%g' % a.ng_timeout) if ng_time is None else '%.2f' % ng_time:>10} "
                  f"{'-' if ng_time is None else '%.1fx' % (ng_time / te):>8} {fmt(t50e) + ' / ' + fmt(t50n):>22} "
                  f"{'-' if worst is None else '%.2g' % worst:>9}{ng_note[:110]}", flush=True)
    finally:
        shutil.rmtree(tmp, True)

if __name__ == "__main__":
    main()
