#!/usr/bin/env python3
"""Differential test: TreeSPICE (acutesim_engine_dump) vs ngspice on the same SPICE netlists.
Stdlib only. Usage: python3 tests/ngspice_diff/run_diff.py [-k substr] [--modes cpu,gpu] [-v]
Exit status is non-zero if any netlist FAILs (or ngspice itself errors)."""
import argparse, atexit, bisect, glob, math, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# The standalone TreeSPICE build, else AcuteSim's build when TreeSPICE is its acutesim_engine/ submodule.
ENGINE = next((p for p in (os.path.join(ROOT, "build", "acutesim_engine_dump"),
                           os.path.join(ROOT, "..", "build", "acutesim_engine", "acutesim_engine_dump"))
               if os.access(p, os.X_OK)), os.path.join(ROOT, "build", "acutesim_engine_dump"))
GROUND = {"0", "GND", "VSS", "GROUND"}
NODE_POS = {**{c: [1, 2] for c in "RCLVIDHF"}, **{c: [1, 2, 3, 4] for c in "MEGT"},
            "Q": [1, 2, 3], "J": [1, 2, 3], "K": []}
SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}

# Tolerance policy (all overridable by flags).
# DC: engine voltage sources are 1 mOhm Norton sources, inductors 1 mOhm shorts, E sources 100 uOhm, so
#   node voltages carry ~I*1mOhm error (<=100 uV at 100 mA); atol=200 uV covers that.  rtol=1e-3 is
#   ngspice's own RELTOL, i.e. the accuracy either solver actually guarantees for nonlinear devices.
# Transient: engine steps are <= the .tran step (coarse, LTE-limited); the reference is run with
#   tmax=tstep/10.  Error is judged against each node's swing (max-min of the reference): 2% of swing,
#   with a 5 mV floor so near-constant nodes (virtual grounds) are not judged on noise.
# ngspice temperature: the engine hardcodes Vt = 25.85 mV (kT/q at 299.98 K = 26.83 C); ngspice defaults to
#   27 C (Vt = 25.865 mV), which alone shifts a 26 mA BJT Vbe by 0.5 mV.  --temp aligns them (set as
#   .options temp= tnom=), so remaining differences are model differences, not a thermal-voltage constant.
DEFAULTS = dict(dc_atol=2e-4, dc_rtol=1e-3, tran_rtol=0.02, tran_atol=5e-3, temp=26.83)

def num(s):
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)([a-z]*)", s.lower())
    if not m: raise ValueError(f"bad number {s!r}")
    suf = m.group(2)
    return float(m.group(1)) * (SUFFIX["meg"] if suf.startswith("meg") else SUFFIX.get(suf[:1], 1.0) if suf else 1.0)

def toks(line): return re.sub(r"[(),=]", " ", line).split()

def cards(text):
    """Yield (index, stripped line) of non-comment lines; '+' continuations are joined to the previous."""
    out = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.split(";")[0].strip()
        if i == 0 or not line or line[0] == "*": continue
        if line[0] == "+" and out: out[-1] += " " + line[1:]
        else: out.append(line)
    return out

def node_map(text):
    """Engine column order: first appearance in top-level element cards (name -> 1-based column)."""
    m, in_sub = {}, False
    for line in cards(text):
        t = toks(line)
        if t[0][0] == ".":
            kw = t[0].upper()
            if kw == ".END": break
            in_sub = kw == ".SUBCKT" or (in_sub and kw != ".ENDS")
            continue
        if in_sub: continue
        c = t[0][0].upper()
        pos = range(1, len(t) - 1) if c == "X" else NODE_POS.get(c, [])
        # Q c b e [s] model ...: the 4th field is a substrate node when the 5th is a name (engine rule)
        if c == "Q" and len(t) >= 6 and not re.fullmatch(r"[-+.\d].*", t[5]) and t[5].upper() not in ("AREA", "OFF", "IC", "M"):
            pos = [1, 2, 3, 4]
        for p in pos:
            if p < len(t) and t[p].upper() not in GROUND: m.setdefault(t[p].upper(), len(m) + 1)
    return m

def analysis(text):
    """-> ('op',) | ('dc', src, start, stop, step) | ('tran', tstep, tstop, uic) | ('ac', sweep, n, f1, f2)
       | ('noise', "v(out[,ref])", src, sweep, n, f1, f2) | ('unsupported', why)"""
    for line in cards(text):
        t = line.split()
        kw = t[0].lower()
        if kw == ".ac": return ("ac", t[1].lower(), int(num(t[2])), num(t[3]), num(t[4]))
        if kw == ".noise":
            k = next(i for i, x in enumerate(t) if x.lower() in ("dec", "oct", "lin"))
            return ("noise", " ".join(t[1:k - 1]), t[k - 1], t[k].lower(), int(num(t[k + 1])), num(t[k + 2]), num(t[k + 3]))
        if kw == ".dc":
            return ("dc", t[1], *map(num, t[2:5])) if len(t) == 5 else ("unsupported", "nested/multi-source .dc")
        if kw == ".tran":
            n = [num(x) for x in t[1:] if x.lower() != "uic"]
            return ("tran", n[0], n[1], any(x.lower() == "uic" for x in t[1:]))
    return ("op",)

# ---------------------------------------------------------------- engine side
def run_engine(args, text, mode, kind, step=0.0, stop=0.0):
    path = os.path.join(args.tmp, "eng.cir")
    open(path, "w").write(text)
    cmd = [args.engine, path, mode, kind] + ([repr(step), repr(stop)] if kind == "tran" else [])
    try: out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout).stdout
    except subprocess.TimeoutExpired: return dict(conv=False, err="TIMEOUT", pts=[])
    conv, err, pts, total = False, "", [], None
    for ln in out.splitlines():
        if ln.startswith("TOTAL"): total = [float(x) for x in ln.split()[1:]]
        elif ln.startswith("CONVERGED"): conv = ln.split()[1] == "1"
        elif ln.startswith("ERROR"): err = ln[6:].strip()
        elif ln and (ln[0].isdigit() or ln[0] == "-"):
            try: pts.append([float(x) for x in ln.split()])
            except ValueError: pass
    return dict(conv=conv, err=err, pts=pts, total=total)

# ---------------------------------------------------------------- ngspice side
def tran_tmax(text, an):
    """Reference max step: tstep/10, and 1/20 of the shortest PULSE edge/width or SIN period.
    With only tstep/10, ngspice crossed a 5 us gate ramp in one 2.5 us step and carried the error
    for the rest of the run (qa_mos_coarse: 114 mV off a dense reference that the engine matched)."""
    spans = []
    for line in cards(text):
        t = toks(line)
        if t[0][0].upper() not in "VI": continue
        u = [x.upper() for x in t]
        try:
            if "PULSE" in u: spans += [num(x) for x in t[u.index("PULSE") + 4:u.index("PULSE") + 7]]  # tr tf pw
            if "SIN" in u and len(t) > u.index("SIN") + 3: spans.append(1 / num(t[u.index("SIN") + 3]))
        except (ValueError, ZeroDivisionError): pass
    edge = min([s for s in spans if s > 0], default=math.inf) / 20
    # ponytail: floor at tstop/1e5 so a ns edge in a ms run stays ~1e5 steps; such a run is checked at that density only.
    return max(min(an[1] / 10, edge), an[2] / 1e5)

def run_ngspice(args, text, an, names):
    """Return (xs, {name: [values]}) for the analysis, or raise RuntimeError. xs = [] for .op.
    Analysis/print cards are removed and replaced by a .control block (batch mode, ngspice defaults);
    the only option set is tmax on .tran so the reference is dense (see tran_tmax)."""
    drop = re.compile(r"\.(op|dc|tran|ac|noise|print|plot|four|end)\b", re.I)
    kept, skip = [], False
    for i, ln in enumerate(text.splitlines()):
        if i > 0 and ln.strip()[:1] == "+":
            if not skip: kept.append(ln)
            continue
        skip = i > 0 and bool(drop.match(ln.strip()))
        if not skip: kept.append(ln)
    cmd = {"op": "op", "dc": "dc {1} {2!r} {3!r} {4!r}".format(*an) if an[0] == "dc" else "",
           "tran": "tran {0!r} {1!r} 0 {2!r}{3}".format(an[1], an[2], tran_tmax(text, an), " uic" if an[3] else "") if an[0] == "tran" else "",
           "ac": "ac {1} {2} {3!r} {4!r}".format(*an) if an[0] == "ac" else "", "noise": ""}[an[0]]
    vecs = " ".join(f"v({n.lower()})" for n in names)
    out = os.path.join(args.tmp, "ng.dat")
    if os.path.exists(out): os.remove(out)
    kept.append(f".options temp={args.temp} tnom={args.temp}")
    ctl = [".control", cmd, f"wrdata {out} {vecs}", "quit", ".endc", ".end"]
    if an[0] == "noise":  # spectra in plot noise1, integrated totals in noise2
        ctl = [".control", "noise {1} {2} {3} {4} {5!r} {6!r}".format(*an), "setplot noise1",
               f"wrdata {out} onoise_spectrum inoise_spectrum", "setplot noise2",
               "print onoise_total inoise_total", "quit", ".endc", ".end"]
    cir = os.path.join(args.tmp, "ng.cir")
    open(cir, "w").write("\n".join(kept + ctl) + "\n")
    p = subprocess.run([args.ngspice, "-b", cir], capture_output=True, text=True, timeout=args.timeout)
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise RuntimeError("ngspice produced no data: " + " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:]))
    rows = [[float(x) for x in ln.split()] for ln in open(out) if ln.strip()]
    if an[0] == "noise":
        tot = dict(re.findall(r"(\w+_total) = (\S+)", p.stdout))
        return [r[0] for r in rows], {"onoise": [r[1] for r in rows], "inoise": [r[3] for r in rows],
                                      "total": [float(tot["onoise_total"]), float(tot["inoise_total"])]}
    # wrdata layout: (scale, v1, scale, v2, ...); complex (.ac): (scale, re, im) per vector.
    # For .op the "scale" is meaningless.
    xs = [] if an[0] == "op" else [r[0] for r in rows]
    if an[0] == "ac":
        return xs, {n: [complex(r[3 * k + 1], r[3 * k + 2]) for r in rows] for k, n in enumerate(names)}
    return xs, {n: [r[2 * k + 1] for r in rows] for k, n in enumerate(names)}

# ---------------------------------------------------------------- comparison
def interp(xs, ys, t):
    i = bisect.bisect_left(xs, t)
    if i == 0: return ys[0]
    if i >= len(xs): return ys[-1]
    if xs[i] == xs[i - 1]: return ys[i]
    w = (t - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + w * (ys[i] - ys[i - 1])

def col(r, idx): return [row[idx] for row in r["pts"]]

def unsupported(r):
    return not r["conv"] and bool(re.search(r"unsupported|not supported|unknown (element|card)", r["err"], re.I))

def compare(args, text, an, mode, nm, ref):
    """-> (status, worst_abs, worst_rel, note)"""
    xs, vals = ref
    names = list(nm)
    worst_abs = worst_rel = 0.0
    bad = []
    def judge(label, got, exp, tol, scale=None):
        # rel = |err| / max(|ref|, 1 mV) for DC;  |err| / node swing for transient (scale)
        nonlocal worst_abs, worst_rel
        e = abs(got - exp) if math.isfinite(got) else math.inf
        worst_abs = max(worst_abs, e); worst_rel = max(worst_rel, e / (scale or max(abs(exp), 1e-3)))
        if not e <= tol: bad.append(f"{label}: {got:.6g} vs ngspice {exp:.6g}")
    if an[0] == "noise":
        r = run_engine(args, text, mode, "noise")
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"] or len(r["pts"]) != len(xs) or not r["total"]:
            return "FAIL", math.inf, math.inf, f"engine gave {len(r['pts'])} of {len(xs)} frequencies: {r['err']}"
        def rel(label, got, exp):  # noise densities and totals: relative (DC rtol)
            nonlocal worst_abs, worst_rel
            e = abs(got - exp); worst_abs = max(worst_abs, e); worst_rel = max(worst_rel, e / max(abs(exp), 1e-300))
            if not e <= args.dc_rtol * abs(exp): bad.append(f"{label}: {got:.6g} vs ngspice {exp:.6g}")
        for k, row in enumerate(r["pts"]):
            rel(f"f={xs[k]:g} onoise", row[1], vals["onoise"][k])
            rel(f"f={xs[k]:g} inoise", row[2], vals["inoise"][k])
        rel("onoise_total", r["total"][0], vals["total"][0])
        rel("inoise_total", r["total"][1], vals["total"][1])
    elif an[0] == "ac":
        r = run_engine(args, text, mode, "ac")
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"] or len(r["pts"]) != len(xs):
            return "FAIL", math.inf, math.inf, f"engine gave {len(r['pts'])} of {len(xs)} frequencies: {r['err']}"
        for k, row in enumerate(r["pts"]):
            if abs(row[0] - xs[k]) > 1e-9 * xs[k]: bad.append(f"f={row[0]:g} vs ngspice {xs[k]:g}")
            for n in names:
                c = 2 * nm[n] - 1
                got, exp = complex(row[c], row[c + 1]), vals[n][k]
                e = abs(got - exp)
                worst_abs = max(worst_abs, e); worst_rel = max(worst_rel, e / max(abs(exp), 1e-3))
                if not e <= args.dc_atol + args.dc_rtol * abs(exp):
                    bad.append(f"f={xs[k]:g} v({n}): {got:.6g} vs ngspice {exp:.6g}")
    elif an[0] in ("op", "dc"):
        r = run_engine(args, text, mode, "dc")   # .op: one row; .dc: one row per sweep point
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"] or not r["pts"]: return "FAIL", math.inf, math.inf, f"engine not converged: {r['err']}"
        if an[0] == "dc" and len(r["pts"]) != len(xs):
            return "FAIL", math.inf, math.inf, f"engine swept {len(r['pts'])} points, ngspice {len(xs)}"
        for k, row in enumerate(r["pts"]):
            at = "" if an[0] == "op" else f"{an[1]}={row[0]:g} "
            if an[0] == "dc" and abs(row[0] - xs[k]) > 1e-9 * max(1.0, abs(xs[k])): bad.append(f"{at}sweep value vs ngspice {xs[k]:g}")
            for n in names:
                exp = vals[n][k]
                judge(f"{at}v({n})", row[nm[n]], exp, args.dc_atol + args.dc_rtol * abs(exp))
    else:
        tstep, tstop = an[1], an[2]
        r = run_engine(args, text, mode, "tran", tstep, tstop)
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["pts"]: return "FAIL", math.inf, math.inf, f"engine produced no waveform: {r['err']}"
        ts = col(r, 0)
        if ts[-1] < tstop * 0.999: bad.append(f"engine stopped at t={ts[-1]:g} < tstop={tstop:g} ({r['err']})")
        sel = [i for i, t in enumerate(ts) if xs[0] <= t <= xs[-1]]  # reference may start at TSTART
        for n in names:
            ref_n = vals[n]
            swing = max(ref_n) - min(ref_n)
            tol = max(args.tran_atol, args.tran_rtol * swing)
            eng = col(r, nm[n])
            for i in sel:
                k = bisect.bisect_left(xs, ts[i])
                if 0 < k < len(xs) and abs(ref_n[k] - ref_n[k - 1]) > tol:
                    # The reference jumps more than tol inside one cell (a switching edge): interpolation
                    # would invent a value, so the engine must lie within the cell's bracket +- tol.
                    lo, hi = sorted((ref_n[k - 1], ref_n[k]))
                    got = min(max(eng[i], lo), hi)
                    judge(f"t={ts[i]:g} v({n})", eng[i], got, tol, max(swing, 1e-3))
                    continue
                judge(f"t={ts[i]:g} v({n})", eng[i], interp(xs, ref_n, ts[i]), tol, max(swing, 1e-3))
        if not r["conv"] and not bad: bad.append(f"engine reports not converged: {r['err']}")
    return ("FAIL" if bad else "PASS"), worst_abs, worst_rel, "; ".join(bad[:3]) + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else "")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-k", default="", help="only netlists whose name contains this substring")
    ap.add_argument("--modes", default="cpu,gpu")
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--ngspice", default="ngspice")
    ap.add_argument("--timeout", type=int, default=300)
    for k, v in DEFAULTS.items(): ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("-v", action="store_true", help="print mismatch details")
    args = ap.parse_args()
    if not os.access(args.engine, os.X_OK):
        sys.exit(f"engine not found: {args.engine}  (cmake -S . -B build && cmake --build build --target acutesim_engine_dump)")
    args.tmp = tempfile.mkdtemp(prefix="ngdiff_")
    atexit.register(shutil.rmtree, args.tmp, True)
    fails = 0
    for path in sorted(glob.glob(os.path.join(HERE, "netlists", "*.cir"))):
        name = os.path.basename(path)
        if args.k not in name: continue
        text = open(path).read()
        an = analysis(text)
        if an[0] == "unsupported":
            print(f"{'SKIPPED':<19} {name} ({an[1]})"); continue
        nm = node_map(text) if an[0] != "noise" else {}
        try: ref = run_ngspice(args, text, an, list(nm))
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"{'NGSPICE-ERROR':<19} {name} {e}"); fails += 1; continue
        for mode in args.modes.split(","):
            st, wa, wr, note = compare(args, text, an, mode, nm, ref)
            fails += st == "FAIL"
            print(f"{st:<19} {name:<32} [{mode}] worst abs={wa:.3g} V  rel={wr:.3g}" + (f"  {note}" if note and (args.v or st != "PASS") else ""), flush=True)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
