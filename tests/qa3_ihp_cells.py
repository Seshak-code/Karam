#!/usr/bin/env python3
"""QA round 3: every IHP SG13G2 standard cell (sg13g2_stdcell.spice) in a transient against ngspice.
    python3 tests/qa3_ihp_cells.py [-k substr] [--modes cpu]
Inputs get pulse trains of different periods (clock-like pins fast, data/enable slow, resets pulse once);
every output sees 5 fF. Waveforms are judged with ngspice_diff's transient rule (2 % of swing / 5 mV)."""
import argparse, os, re, shutil, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ngspice_diff"))
import run_diff as rd  # noqa: E402

IHP = "pdk/ihp/ihp-sg13g2"
OUTS = {"X", "Y", "Q", "Q_N", "Z", "GCLK", "SH", "L_HI", "L_LO"}
V = 1.2

def drive(pin, k):
    p = pin.upper()
    if p in ("CLK", "GATE", "GATE_N"): return f"PULSE(0 {V} 100p 40p 40p 450p 1n)"
    if p in ("RESET_B", "SET_B"): return f"PULSE({V} 0 3n 20p 20p 300p 20n)"
    if p == "TE_B": return f"PULSE(0 {V} 2.5n 30p 30p 1.5n 4n)"
    per = [1.3, 2.1, 3.4, 5.5, 1.7, 2.9][k % 6]
    return f"PULSE(0 {V} {0.2 + 0.1 * k}n 40p 40p {per / 2}n {per}n)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("--modes", default="cpu")
    a = ap.parse_args()
    root = os.path.dirname(HERE)
    args = types.SimpleNamespace(**rd.DEFAULTS, engine=rd.ENGINE, ngspice="ngspice", timeout=300,
                                 tmp=tempfile.mkdtemp(prefix="ihpc_"), v=False)
    os.symlink(os.path.join(root, "datasets", "pdk"), os.path.join(args.tmp, "pdk"))
    cells = re.findall(r"(?im)^\.subckt\s+(\S+)\s+(.*)$", open(os.path.join(root, "datasets", "pdk", "ihp", "ihp-sg13g2",
                       "libs.ref", "sg13g2_stdcell", "spice", "sg13g2_stdcell.spice")).read())
    bad = 0
    for name, pins in cells:
        if a.k not in name: continue
        pins = pins.split()
        sig = [p for p in pins if p.upper() not in ("VDD", "VSS")]
        ins = [p for p in sig if p.upper() not in OUTS]
        outs = [p for p in sig if p.upper() in OUTS]
        if not outs: print(f"SKIP {name} (no output)"); continue
        body = "".join(f"V{p} n_{p} 0 {drive(p, k)}\n" for k, p in enumerate(ins))
        body += "".join(f"CL{p} n_{p} 0 5f\n" for p in outs)
        conn = " ".join("vdd" if p.upper() == "VDD" else "0" if p.upper() == "VSS" else f"n_{p}" for p in pins)
        text = (f"* {name}\n.osdi psp103.osdi\n.lib {IHP}/libs.tech/ngspice/models/cornerMOSlv.lib mos_tt\n"
                f".include {IHP}/libs.ref/sg13g2_stdcell/spice/sg13g2_stdcell.spice\nVDD vdd 0 DC {V}\n{body}X1 {conn} {name}\n"
                ".tran 20p 8n\n.end\n")
        an, nm = rd.analysis(text), rd.node_map(text)
        try: ref = rd.run_ngspice(args, text, an, list(nm))
        except Exception as e: print(f"NGSPICE-ERROR {name}: {str(e)[:100]}"); continue
        for mode in a.modes.split(","):
            st, wa, wr, note = rd.compare(args, text, an, mode, nm, ref)
            bad += st == "FAIL"
            print(f"{st:<19} {name:<24} [{mode}] worst abs={wa:.3g} V rel={wr:.3g} {note[:200] if st != 'PASS' else ''}", flush=True)
    shutil.rmtree(args.tmp, True)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
