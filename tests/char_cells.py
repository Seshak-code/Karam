#!/usr/bin/env python3
"""Standard-cell characterisation against ngspice on a real PDK (Roadmap Phase 3 gate).

    python3 tests/char_cells.py [-k inv] [--tol 0.01]

For IHP SG13G2 cells (PSP103 through OSDI) and sky130 cells built from its devices (native BSIM4
cards, run by the engine on the OSDI BSIM4), tt corner, both simulators run the same transient:
the input switches, the output drives a load, VDD comes through a 1 ohm sense resistor. Compared:
every propagation delay (input 50 % -> output 50 %), every output slew (20 -> 80 %) and the supply
energy over the run. Exit 1 if any differs from ngspice by more than --tol (default 1 %).
Needs datasets/pdk/{ihp,sky130} (see ENGINE_HANDOFF.md) and datasets/osdi (tests/build_osdi.sh); a PDK
that is missing is skipped.
"""
import argparse, os, shutil, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ngspice_diff"))
import run_diff as rd  # noqa: E402

RS = 1.0
IHP = "pdk/ihp/ihp-sg13g2"
SKY_CELLS = """.subckt sky_inv a y vpwr vgnd
XP y a vpwr vpwr sky130_fd_pr__pfet_01v8_hvt l=0.15 w=1.0
XN y a vgnd vgnd sky130_fd_pr__nfet_01v8 l=0.15 w=0.65
.ends
.subckt sky_nand2 a b y vpwr vgnd
XP1 y a vpwr vpwr sky130_fd_pr__pfet_01v8_hvt l=0.15 w=1.0
XP2 y b vpwr vpwr sky130_fd_pr__pfet_01v8_hvt l=0.15 w=1.0
XN1 y a m vgnd sky130_fd_pr__nfet_01v8 l=0.15 w=0.65
XN2 m b vgnd vgnd sky130_fd_pr__nfet_01v8 l=0.15 w=0.65
.ends
.subckt sky_nor2 a b y vpwr vgnd
XP1 m a vpwr vpwr sky130_fd_pr__pfet_01v8_hvt l=0.15 w=1.0
XP2 y b m vpwr sky130_fd_pr__pfet_01v8_hvt l=0.15 w=1.0
XN1 y a vgnd vgnd sky130_fd_pr__nfet_01v8 l=0.15 w=0.65
XN2 y b vgnd vgnd sky130_fd_pr__nfet_01v8 l=0.15 w=0.65
.ends
"""
# pdk -> (VDD, header after the supply, needed paths, {cell: (circuit, trigger, output, tstep, tstop)})
def pdks():
    out = {}
    for name, vdd, head, need, cells in (
        ("ihp", 1.2, f""".osdi psp103.osdi
.lib {IHP}/libs.tech/ngspice/models/cornerMOSlv.lib mos_tt
.include {IHP}/libs.ref/sg13g2_stdcell/spice/sg13g2_stdcell.spice
""", [IHP, "osdi/psp103.osdi"], {
            "inv_1": ("X1 y a vdd 0 sg13g2_inv_1", "DC {v}", "a", "y", 2e-9),
            "nand2_1": ("X1 y a b vdd 0 sg13g2_nand2_1", "DC {v}", "a", "y", 2e-9),
            "nor2_1": ("X1 y a b vdd 0 sg13g2_nor2_1", "DC 0", "a", "y", 2e-9),
            "xor2_1": ("X1 y a b vdd 0 sg13g2_xor2_1", "DC {v}", "a", "y", 2e-9),
            "dfrbp_1": ("VC clk 0 PULSE(0 {v} 200p 40p 40p 460p 1n)\nVR rb 0 PULSE(0 {v} 50p 10p 10p 10n 20n)\n"
                        "X1 q qn clk a rb vdd 0 sg13g2_dfrbp_1", None, "clk", "q", 4e-9)}),
        ("sky130", 1.8, ".lib lib/sky130_tt_min.lib tt\n" + SKY_CELLS, ["pdk/sky130", "osdi/bsim4.osdi"], {
            "inv": ("X1 a y vdd 0 sky_inv", "DC {v}", "a", "y", 2e-9),
            "nand2": ("X1 a b y vdd 0 sky_nand2", "DC {v}", "a", "y", 2e-9),
            "nor2": ("X1 a b y vdd 0 sky_nor2", "DC 0", "a", "y", 2e-9)})):
        built = {}
        for cell, (inst, vb, trig, outn, stop) in cells.items():
            # input a toggles (the flip-flop's D: slower), b holds the other input non-controlling
            a = f"PULSE(0 {vdd} 0 40p 40p 900p 2n)" if vb is None else f"PULSE(0 {vdd} 100p 40p 40p 400p 1n)"
            body = f"VA a 0 {a}\n" + (f"VB b 0 {vb.format(v=vdd)}\n" if vb else "") + inst.format(v=vdd) + f"\nCL {outn} 0 5f"
            built[f"{name}:{cell}"] = (vdd, f"VS vs 0 DC {vdd}\nRS vs vdd {RS}\n" + head, body, trig, outn, 0.5e-12, stop)
        out[name] = (need, built)
    return out

def crossings(ts, vs, level):
    out = []
    for i in range(1, len(ts)):
        a, b = vs[i - 1] - level, vs[i] - level
        if a == 0 or a * b < 0:
            out.append((ts[i - 1] + (level - vs[i - 1]) * (ts[i] - ts[i - 1]) / (vs[i] - vs[i - 1]), b > a))
    return out

def metrics(ts, w, trig, out, VDD):
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
    args = types.SimpleNamespace(**rd.DEFAULTS, engine=a.engine, ngspice="ngspice", timeout=600,
                                 tmp=tempfile.mkdtemp(prefix="char_"), v=False)
    os.symlink(os.path.join(root, "datasets", "pdk"), os.path.join(args.tmp, "pdk"))
    shutil.copytree(os.path.join(HERE, "ngspice_diff", "netlists", "lib"), os.path.join(args.tmp, "lib"))
    cells = {}
    for pdk, (need, built) in pdks().items():
        missing = [p for p in need if not os.path.exists(os.path.join(root, "datasets", p))]
        if missing: print(f"SKIP  {pdk}: missing datasets/{', datasets/'.join(missing)}"); continue
        cells.update(built)
    worst, bad = 0.0, 0
    try:
        for name, (VDD, head, body, trig, out, step, stop) in cells.items():
            if a.k not in name: continue
            text = f"* char {name}\n{head}{body}\n.tran {step!r} {stop!r}\n.end\n"
            an = rd.analysis(text)
            nm = rd.node_map(text)
            xs, ref = rd.run_ngspice(args, text, an, list(nm))
            r = rd.run_engine(args, text, "cpu", "tran", step, stop)
            if not r["conv"]:
                print(f"FAIL  {name:<14} engine: {r['err'][:150]}"); bad += 1; continue
            eng = {n.lower(): [row[c] for row in r["pts"]] for n, c in nm.items()}
            me = metrics([row[0] for row in r["pts"]], eng, trig, out, VDD)
            mr = metrics(xs, {k.lower(): v for k, v in ref.items()}, trig, out, VDD)
            if set(me) != set(mr):
                print(f"FAIL  {name:<14} transitions differ: engine {sorted(me)} vs ngspice {sorted(mr)}"); bad += 1; continue
            errs = {k: abs(me[k] - mr[k]) / abs(mr[k]) for k in mr}
            k = max(errs, key=errs.get)
            worst = max(worst, errs[k])
            st = "PASS" if errs[k] <= a.tol else "FAIL"
            bad += st == "FAIL"
            detail = "  ".join(f"{x}={me[x]*1e12:.2f}ps({errs[x]*100:.2f}%)" for x in sorted(me) if x != "energy")
            print(f"{st}  {name:<14} energy={me['energy']*1e15:.1f}fJ({errs['energy']*100:.2f}%)  {detail}", flush=True)
    finally:
        shutil.rmtree(args.tmp, True)
    print(f"worst {worst*100:.3f} % (tolerance {a.tol*100:g} %)")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
