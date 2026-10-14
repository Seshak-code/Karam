#!/usr/bin/env python3
"""TreeSPICE contraction solver vs SuiteSparse KLU on SuiteSparse circuit matrices.

Downloads each matrix once into datasets/suitesparse/ (git-ignored), runs build/klu_bench for KLU and
for TreeSPICE (1 thread, and all threads) in separate processes, and prints a Markdown table.
Stdlib only.  Usage: python3 bench/klu_bench.py [-k substr] [--timeout S] [--reps N] [--out FILE]

Columns: fs = factor + solve per Newton iteration (KLU: klu_refactor + klu_solve; TreeSPICE: factor +
solve, no refinement); ratio = TreeSPICE fs (1 thread) / KLU fs; flops = factorization flops;
omega = componentwise backward error (TreeSPICE after its 2 refinement steps, KLU without refinement).
"""
import argparse, os, re, subprocess, sys, tarfile, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "suitesparse")
URL = "https://sparse.tamu.edu/MM/{}.tar.gz"
# Circuit-simulation matrices used in the KLU paper's class (Davis & Palamadai Natarajan, TOMS 2010),
# smallest first; AMD/G2_circuit is a grid-like (SPD) control where nested dissection should do well.
MATRICES = ["Bomhof/circuit_1", "Hamm/add20", "Bomhof/circuit_2", "Hamm/add32", "Grund/meg4",
            "Rajat/rajat03", "Bomhof/circuit_3", "Hamm/memplus", "Rajat/rajat15", "Hamm/bcircuit",
            "Bomhof/circuit_4", "Sandia/ASIC_100k", "Hamm/hcircuit", "IBM_EDA/dc1", "IBM_EDA/trans4",
            "AMD/G2_circuit", "Hamm/scircuit", "Freescale/transient", "Sandia/ASIC_320k", "Rajat/rajat21",
            "Rajat/rajat30", "Sandia/ASIC_680k", "Freescale/memchip"]


def fetch(name):
    group, base = name.split("/")
    mtx = os.path.join(DATA, base, base + ".mtx")
    if not os.path.exists(mtx):
        os.makedirs(DATA, exist_ok=True)
        tgz = os.path.join(DATA, base + ".tar.gz")
        print(f"  downloading {name} ...", file=sys.stderr, flush=True)
        urllib.request.urlretrieve(URL.format(name), tgz)
        with tarfile.open(tgz) as tf:
            tf.extract(f"{base}/{base}.mtx", DATA, filter="data")
        os.remove(tgz)
    return mtx


def run(exe, mtx, solver, reps, timeout, threads=None):
    env = dict(os.environ)
    if threads: env["TREESPICE_THREADS"] = str(threads)
    try:
        out = subprocess.run([exe, mtx, solver, str(reps)], capture_output=True, text=True,
                             timeout=timeout, env=env).stdout
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    return {k: v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', out)}


def ms(d, k):
    if d.get("timeout"): return "timeout"
    return f"{float(d[k]) * 1e3:.3g}" if k in d else d.get("error", "fail")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-k", default="")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "klu_bench"))
    a = ap.parse_args()
    if not os.access(a.exe, os.X_OK):
        sys.exit(f"{a.exe} not built (needs KLU: brew install suite-sparse / apt install libsuitesparse-dev)")
    rows = ["| matrix | n | nnz | KLU fs ms | TreeSPICE fs ms (1T) | (all T) | ratio | KLU flops | TreeSPICE flops "
            "| peak front | KLU omega | TreeSPICE omega |", "|" + "---|" * 12]
    for name in [m for m in MATRICES if a.k in m]:
        mtx = fetch(name)
        k = run(a.exe, mtx, "klu", a.reps, a.timeout)
        t1 = run(a.exe, mtx, "tree", a.reps, a.timeout, threads=1)
        tn = run(a.exe, mtx, "tree", a.reps, a.timeout) if not t1.get("timeout") else {"timeout": True}
        ratio = f"{float(t1['fs']) / float(k['fs']):.3g}x" if "fs" in t1 and "fs" in k else "-"
        row = (f"| {name} | {k.get('n', t1.get('n', '?'))} | {k.get('nnz', t1.get('nnz', '?'))} | {ms(k, 'fs')} | "
               f"{ms(t1, 'fs')} | {ms(tn, 'fs')} | {ratio} | {k.get('flops', '-')} | {t1.get('flops', '-')} | "
               f"{t1.get('peak_front', '-')} | {k.get('omega', '-')} | {t1.get('omega', '-')} |")
        rows.append(row)
        print(row, flush=True)
    if a.out:
        open(a.out, "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
