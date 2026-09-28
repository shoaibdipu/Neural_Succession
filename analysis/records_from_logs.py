#!/usr/bin/env python3
"""Convert the console output of three succession scripts into the JSON records used by the analysis.

E3_v1.py, E4_v3.py and E5.py print their per-transition tables and summary statistics to stdout.
Capture that output (e.g. `python succession/E4_v3.py | tee e4_stdout.txt`) and run

    python analysis/records_from_logs.py --e3v1 e3v1_stdout.txt --e4 e4_stdout.txt --e5 e5_stdout.txt --out records/succession

to obtain e3v1_results.json, e4_summary.json and e5_summary.json in the format read by make_figures_and_stats.py.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ARROW = "\u2192"


def parse_e3v1(text: str) -> dict:
    pat = re.compile(r"T(\d)" + ARROW + r"T(\d) \(([^)]*)\)\s+\u03c1_pre=([+-][0-9.]+)\s+forget=([0-9.]+)\s+(\u2713 coexist|\u2717 forget)")
    rows = [dict(tA=int(m.group(1)), tB=int(m.group(2)), classes=m.group(3), rho_pre=float(m.group(4)),
                 forgetting=float(m.group(5)), coexist=int("coexist" in m.group(6))) for m in pat.finditer(text)]
    thr = re.search(r"Coexistence threshold\s*:\s*([0-9.]+)", text)
    return {"script": "succession/E3_v1.py", "coexistence_threshold": float(thr.group(1)) if thr else None, "rows": rows}


def parse_e4(text: str) -> dict:
    t1 = re.compile(r"^\s*(T\d" + ARROW + r"T\d)\s+([+-][0-9.]+)\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)\s*$", re.M)   # pair rho R_AA M* M*_norm
    t2 = re.compile(r"^\s*(T\d" + ARROW + r"T\d)\s+([+-][0-9.]+)\s+([0-9.]+)\s+(-?[0-9.]+)\s+(\d+)\s*$", re.M)  # pair rho F0 eta M*(fix)
    head = text.index("Replay efficiency")
    first = {m.group(1): m for m in t1.finditer(text[:head])}
    second = {m.group(1): m for m in t2.finditer(text[head:])}
    rows = []
    for pair, m in first.items():
        s = second[pair]
        rows.append({"pair": pair, "rho_pre": float(m.group(2)), "R_AA": float(m.group(3)), "M_star": int(m.group(4)),
                     "M_star_norm": float(m.group(5)), "forgetting_M0": float(s.group(3)), "eta_M0_to_M25": float(s.group(4)),
                     "M_star_fix": int(s.group(5))})
    raw = re.search(r"--- Raw M\* ---\s*\n\s*Pearson r : ([+-]?[0-9.]+)\s+p=([0-9.]+)\s*\n\s*Spearman \u03c1: ([+-]?[0-9.]+)", text)
    eff = re.search(r"Pearson r\s+\(\u03c1_pre vs \u03b7\): ([+-]?[0-9.]+)\s+p=([0-9.]+)\s*\n\s*Spearman \u03c1 \(\u03c1_pre vs \u03b7\): ([+-]?[0-9.]+)", text)
    thr = re.search(r"Threshold\s*:\s*(.+)", text)
    return {"setting": "Split-CIFAR-10 / ResNet-18 / 20 directed pairs / replay buffers 0..1000",
            "threshold": thr.group(1).strip().replace("\u00d7", "x") if thr else "0.1 x R_AA",
            "raw_Mstar": {"pearson_r": float(raw.group(1)), "p": float(raw.group(2)), "spearman_rho": float(raw.group(3))},
            "initial_efficiency": {"pearson_r": float(eff.group(1)), "p": float(eff.group(2)), "spearman_rho": float(eff.group(3))},
            "script": "succession/E4_v3.py", "rows": rows}


def parse_e5(text: str) -> dict:
    seeds = re.search(r"Seeds\s*:\s*(\d+)", text)
    pat = re.compile(r"^\s*([a-z_]+)\s+([0-9.]+)\u00b1([0-9.]+)\s+(-?[0-9.]+)\s+([0-9.]+)\u00b1([0-9.]+)\s+(-?[0-9.]+)\s*$", re.M)
    table = text[text.rindex("Method"):]
    res = {}
    for m in pat.finditer(table):
        res[m.group(1)] = {"cifar10": {"ACC_mean": float(m.group(2)), "ACC_sd": float(m.group(3)), "BWT": float(m.group(4))},
                           "cifar100": {"ACC_mean": float(m.group(5)), "ACC_sd": float(m.group(6)), "BWT": float(m.group(7))}}
    return {"seeds": int(seeds.group(1)) if seeds else None, "script": "succession/E5.py", "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3v1"); ap.add_argument("--e4"); ap.add_argument("--e5")
    ap.add_argument("--out", default="records/succession")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for arg, fn, name in [(a.e3v1, parse_e3v1, "e3v1_results.json"), (a.e4, parse_e4, "e4_summary.json"), (a.e5, parse_e5, "e5_summary.json")]:
        if arg:
            rec = fn(Path(arg).read_text(encoding="utf-8", errors="replace"))
            json.dump(rec, open(out / name, "w"), indent=1, ensure_ascii=False)
            print("wrote", out / name)


if __name__ == "__main__":
    main()
