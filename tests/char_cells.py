#!/usr/bin/env python3
"""Standard-cell characterisation against ngspice on a real PDK (Roadmap Phase 3 gate).

    python3 tests/char_cells.py [-k inv] [--tol 0.01]

For IHP SG13G2 cells (PSP103 through OSDI, tt corner) both simulators run the same transient:
the input switches, the output drives a load, VDD comes through a 1 ohm sense resistor. Compared:
every propagation delay (input 50 % -> output 50 %), every output slew (20 -> 80 %) and the supply
energy over the run. Exit 1 if any differs from ngspice by more than --tol (default 1 %).
Needs datasets/pdk/ihp (see ENGINE_HANDOFF.md) and datasets/osdi/psp103.osdi (tests/build_osdi.sh).
"""
import argparse, os, shutil, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ngspice_diff"))
import run_diff as rd  # noqa: E402

PDK = "pdk/ihp/ihp-sg13g2"
VDD, RS = 1.2, 1.0
HEAD = f""".osdi psp103.osdi
.lib {PDK}/libs.tech/ngspice/models/cornerMOSlv.lib mos_tt
.include {PDK}/libs.ref/sg13g2_stdcell/spice/sg13g2_stdcell.spice
VS vs 0 DC {VDD}
RS vs vdd {RS}
"""
PULSE = f"PULSE(0 {VDD} 100p 40p 40p 400p 1n)"
# name -> (circuit lines, trigger node, output node, tstep, tstop)
CELLS = {
    "inv_1": (f"VA a 0 {PULSE}\nX1 y a vdd 0 sg13g2_inv_1\nCL y 0 5f", "a", "y", 0.5e-12, 2e-9),
    "nand2_1": (f"VA a 0 {PULSE}\nVB b 0 DC {VDD}\nX1 y a b vdd 0 sg13g2_nand2_1\nCL y 0 5f", "a", "y", 0.5e-12, 2e-9),
    "nor2_1": (f"VA a 0 {PULSE}\nVB b 0 DC 0\nX1 y a b vdd 0 sg13g2_nor2_1\nCL y 0 5f", "a", "y", 0.5e-12, 2e-9),
    "xor2_1": (f"VA a 0 {PULSE}\nVB b 0 DC {VDD}\nX1 y a b vdd 0 sg13g2_xor2_1\nCL y 0 5f", "a", "y", 0.5e-12, 2e-9),
    "dfrbp_1": (f"VC clk 0 PULSE(0 {VDD} 200p 40p 40p 460p 1n)\nVD d 0 PULSE(0 {VDD} 0 40p 40p 900p 2n)\n"
                f"VR rb 0 PULSE(0 {VDD} 50p 10p 10p 10n 20n)\nX1 q qn clk d rb vdd 0 sg13g2_dfrbp_1\nCL q 0 5f", "clk", "q", 0.5e-12, 4e-9),
}

def crossings(ts, vs, level):
    out = []
    for i in range(1, len(ts)):
        a, b = vs[i - 1] - level, vs[i] - level
        if a == 0 or a * b < 0:
            out.append((ts[i - 1] + (level - vs[i - 1]) * (ts[i] - ts[i - 1]) / (vs[i] - vs[i - 1]), b > a))
    return out

def metrics(ts, w, trig, out):
    """-> {name: value}: delay_k, slew_k for each output transition, and supply energy."""
    m = {}
    tin = [t for t, _ in crossings(ts, w[trig], VDD / 2)]
    rising_trig = [t for t, up in crossings(ts, w[trig], VDD / 2) if up]
    lo, hi = crossings(ts, w[out], 0.2 * VDD), crossings(ts, w[out], 0.8 * VDD)
    for k, (t, up) in enumerate(crossings(ts, w[out], VDD / 2)):
        src = rising_trig if trig == "clk" else tin  # flip-flop: from the clock's rising edge
        before = [x for x in src if x < t]
        if before: m[f"delay{k}{'r' if up else 'f'}"] = t - before[-1]
        a = [x for x, u in (lo if up else hi) if u == up and x < t]
        b = [x for x, u in (hi if up else lo) if u == up and x > t]
        if a and b: m[f"slew{k}{'r' if up else 'f'}"] = b[0] - a[-1]
    i = [(VDD - v) / RS for v in w["vdd"]]
    m["energy"] = VDD * sum((ts[j] - ts[j - 1]) * (i[j] + i[j - 1]) / 2 for j in range(1, len(ts)))
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--engine", default=rd.ENGINE)
    a = ap.parse_args()
    root = os.path.dirname(HERE)
    need = [os.path.join(root, "datasets", PDK), os.path.join(rd.OSDI_DIR, "psp103.osdi")]
    if not all(os.path.exists(p) for p in need):
        sys.exit("missing " + " or ".join(p for p in need if not os.path.exists(p)))
    args = types.SimpleNamespace(**rd.DEFAULTS, engine=a.engine, ngspice="ngspice", timeout=600,
                                 tmp=tempfile.mkdtemp(prefix="char_"), v=False)
    os.symlink(os.path.join(root, "datasets", "pdk"), os.path.join(args.tmp, "pdk"))
    worst, bad = 0.0, 0
    try:
        for name, (body, trig, out, step, stop) in CELLS.items():
            if a.k not in name: continue
            text = f"* char {name}\n{HEAD}{body}\n.tran {step!r} {stop!r}\n.end\n"
            an = rd.analysis(text)
            nm = rd.node_map(text)
            xs, ref = rd.run_ngspice(args, text, an, list(nm))
            r = rd.run_engine(args, text, "cpu", "tran", step, stop)
            if not r["conv"]:
                print(f"FAIL  {name:<9} engine: {r['err'][:150]}"); bad += 1; continue
            eng = {n.lower(): [row[c] for row in r["pts"]] for n, c in nm.items()}
            me = metrics([row[0] for row in r["pts"]], eng, trig, out)
            mr = metrics(xs, {k.lower(): v for k, v in ref.items()}, trig, out)
            if set(me) != set(mr):
                print(f"FAIL  {name:<9} transitions differ: engine {sorted(me)} vs ngspice {sorted(mr)}"); bad += 1; continue
            errs = {k: abs(me[k] - mr[k]) / abs(mr[k]) for k in mr}
            k = max(errs, key=errs.get)
            worst = max(worst, errs[k])
            st = "PASS" if errs[k] <= a.tol else "FAIL"
            bad += st == "FAIL"
            detail = "  ".join(f"{x}={me[x]*1e12:.2f}ps({errs[x]*100:.2f}%)" for x in sorted(me) if x != "energy")
            print(f"{st}  {name:<9} energy={me['energy']*1e15:.1f}fJ({errs['energy']*100:.2f}%)  {detail}", flush=True)
    finally:
        shutil.rmtree(args.tmp, True)
    print(f"worst {worst*100:.3f} % (tolerance {a.tol*100:g} %)")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
