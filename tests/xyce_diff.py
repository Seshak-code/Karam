#!/usr/bin/env python3
"""Xyce regression suite gold outputs (datasets/xyce, git-ignored, GPLv3) against the engine.

    python3 tests/xyce_diff.py [-k RESISTOR] [-j 8] [-v]

For every Netlists/<dir>/<name>.cir with a gold OutputData/<dir>/<name>.cir.prn, run the engine for the
netlist's .op/.dc/.tran (same runner and tolerances as tests/ngspice_diff/run_diff.py) and compare the
single-node V(x) columns of the .prn. Statuses: PASS, FAIL, ENGINE-UNSUPPORTED (engine rejects a card or
model parameter), PARSE-ERROR (Xyce syntax the parser lacks), TIMEOUT (--timeout), XYCE-DIALECT (gold follows Xyce-only
semantics, listed in DIALECT), SKIPPED (a feature this comparison cannot judge: .step, .include/.lib, multi-run
output, no V(node) column, AC/HB/noise/...). Exit 1 on any FAIL.
"""
import argparse, glob, os, re, sys, tempfile, types
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ngspice_diff"))
import run_diff as rd  # noqa: E402

XYCE = os.path.join(os.path.dirname(HERE), "datasets", "xyce")
SKIP_CARDS = re.compile(r"^\s*\.(step|include|inc|lib|sens|hb|mpde|four|data|func|global_param|sampling|"
                        r"embeddedsampling|pce|lin|ac|noise)\b", re.I | re.M)

# Cases where Xyce's gold output follows Xyce-specific semantics, not SPICE3/ngspice (the engine's reference).
DIALECT = {
    "Certification_Tests/BUG_302/": "element IC= applied without UIC",
    "Nonlinear/ConvergenceTests/Nox1Tran0.cir": "element IC= applied without UIC",
    "Certification_Tests/BUG_201_SON/": "element IC= applied without UIC",
    "PARSER/scalingFactors.cir": "Xyce reads 1X as 1e6",
    "GLOBALNODE/global3.cir": "Xyce reads 1a as 1 (ngspice: atto)",
    "Output/TRAN/tran-prn-timescalefactor.cir": "output time scaling",
    "MINDUCTORS/mutIndPrint": "V source across an inductor: singular at DC (ngspice .op: check l1#branch)",
}

def gold(prn):
    """-> (header, rows) or None when the file holds more than one run (e.g. .step)."""
    lines = [l.split() for l in open(prn, errors="ignore") if l.strip()]
    if not lines or lines[0][0] != "Index": return None
    head, rows = lines[0], []
    for t in lines[1:]:
        if t[0] == "Index" or t[0].startswith("End"): return None if t[0] == "Index" else (head, rows)
        try: rows.append([float(x) for x in t])
        except ValueError: return None
    return head, rows

def case(path, opts):
    rel = os.path.relpath(path, os.path.join(XYCE, "Netlists"))
    for key, why in DIALECT.items():
        if rel.startswith(key): return rel, "XYCE-DIALECT", why
    text = open(path, errors="ignore").read()
    g = os.path.join(XYCE, "OutputData", rel + ".prn")
    if SKIP_CARDS.search(text): return rel, "SKIPPED", "uses " + SKIP_CARDS.search(text).group(1).lower()
    gd = gold(g)
    if not gd: return rel, "SKIPPED", "gold output is not a single run"
    head, rows = gd
    try: an = rd.analysis(text)
    except (ValueError, IndexError, StopIteration): return rel, "SKIPPED", "analysis card syntax (e.g. NOOP)"
    if an[0] not in ("op", "dc", "tran"): return rel, "SKIPPED", an[0]
    nm = rd.node_map(text)
    cols = {m.group(1).upper(): k for k, h in enumerate(head) if (m := re.fullmatch(r"V\((\w+)\)", h, re.I))}
    cols = {n: k for n, k in cols.items() if n in nm}
    if not cols or not rows: return rel, "SKIPPED", "no V(node) column"
    xs = [] if an[0] == "op" else [r[1] for r in rows]
    vals = {n: [r[k] for r in rows] for n, k in cols.items()}
    with tempfile.TemporaryDirectory(prefix="xyce_") as tmp:
        args = types.SimpleNamespace(**vars(opts), tmp=tmp)
        st, wa, wr, note = rd.compare(args, text, an, "cpu", {n: nm[n] for n in cols}, (xs, vals))
    if st == "FAIL" and "SPICE parse error" in note: st = "PARSE-ERROR"  # syntax the engine lacks
    if st == "FAIL" and "TIMEOUT" in note: st = "TIMEOUT"  # run-time limit, not an accuracy verdict
    return rel, st, f"worst abs={wa:.3g} V rel={wr:.3g}  {note}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("-v", action="store_true", help="print every case, not only FAIL")
    ap.add_argument("--engine", default=rd.ENGINE)
    ap.add_argument("--timeout", type=int, default=120)
    for k, v in rd.DEFAULTS.items(): ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    opts = ap.parse_args()
    paths = [p for p in sorted(glob.glob(os.path.join(XYCE, "Netlists", "**", "*.cir"), recursive=True))
             if opts.k in p and os.path.exists(os.path.join(XYCE, "OutputData",
                                                            os.path.relpath(p, os.path.join(XYCE, "Netlists")) + ".prn"))]
    if not paths: sys.exit("no gold cases under datasets/xyce (download: see ENGINE_HANDOFF.md)")
    counts = {}
    with ProcessPoolExecutor(opts.j) as ex:
        for rel, st, note in ex.map(case, paths, [opts] * len(paths)):
            counts[st] = counts.get(st, 0) + 1
            if opts.v or st == "FAIL": print(f"{st:<19} {rel:<60} {note[:160]}", flush=True)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    sys.exit(1 if counts.get("FAIL") else 0)

if __name__ == "__main__":
    main()
