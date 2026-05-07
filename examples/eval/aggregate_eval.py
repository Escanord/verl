#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Aggregate eval results from eval_via_verl.sh / eval_<method>_<size>.sh.

Each verl eval run writes a JSONL at:
    <EVAL_OUT_ROOT>/<tag>/drift_eval/<tag>.jsonl

with one row per validation pass, each row carrying val-core/<data_source>/...
keys.  Tags follow the convention <method>_<size>_step<N>:
    v18b_1p7b_step120, grpo_4b_step200, heg_1p7b_step100, v18c_4b_step60, ...

Output formats:
    --format text      pretty pivot table (default; for terminal viewing)
    --format md        markdown tables shaped to drop into the experiment
                       section of docs/drift_method.md
    --format csv       single tidy CSV (run, step, benchmark, metric, value)
    --format jsonl     one row per (method, size, step) with all benchmark
                       metrics flattened into the row.  Use --method/--size
                       to write one summary JSONL per method-model combo.

Usage:
    python aggregate_eval.py --root /path/to/eval_out
    python aggregate_eval.py --root ... --format md > tables.md
    python aggregate_eval.py --root ... --format csv --out summary.csv
    # Per method-model summary jsonl (used by eval_<method>_<size>.sh):
    python aggregate_eval.py --root ... --method v18b --size 1p7b \
        --format jsonl --out v18b_1p7b_summary.jsonl
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Map verl's data_source strings -> short benchmark names used in the doc.
DATA_SOURCE_TO_BENCH = {
    "math__aime_repeated_8x": "AIME24",
    "math__aime25":           "AIME25",
    "math__olympiadbench_en": "OlympiadBench",
    "math__minerva":          "Minerva",
    "mcq__gpqa_diamond":      "GPQA",
}

BENCH_ORDER = ["AIME24", "AIME25", "OlympiadBench", "GPQA", "Minerva"]
METHOD_ORDER = ["v18b", "v18c", "drift", "grpo", "heg"]    # v18b/v18c shown as DRIFT
METHOD_DISPLAY = {"v18b": "DRIFT", "v18c": "DRIFT", "drift": "DRIFT", "grpo": "GRPO", "heg": "HEG"}

# tag = <method>_<size>_step<N>;  e.g. v18b_1p7b_step120, grpo_4b_step200
TAG_RE = re.compile(r"^(?P<method>[a-z0-9]+)_(?P<size>1p7b|4b)_step(?P<step>\d+)$")


def parse_jsonl(path: Path):
    """Return [(step, data_dict), ...] for rows that contain val-core metrics."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            d = row.get("data", row)
            if any(k.startswith("val-core/") for k in d):
                rows.append((row.get("step", -1), d))
    return rows


def extract_metrics(d: dict):
    """Return {bench: {'mean@16': float, 'best@16': float}}."""
    out = defaultdict(dict)
    for k, v in d.items():
        if not k.startswith("val-core/"):
            continue
        body = k[len("val-core/") :]
        parts = body.split("/")
        if len(parts) < 3 or parts[1] != "acc":
            continue
        ds = parts[0]
        bench = DATA_SOURCE_TO_BENCH.get(ds)
        if not bench:
            continue
        if parts[2] == "mean@16" and len(parts) == 3:
            out[bench]["mean@16"] = float(v)
        elif parts[2] == "best@16" and len(parts) >= 4 and parts[3] == "mean":
            out[bench]["best@16"] = float(v)
    return out


def parse_tag(tag: str):
    m = TAG_RE.match(tag)
    if not m:
        return None
    return m.group("method"), m.group("size"), int(m.group("step"))


def discover(root: Path):
    """Yield (tag, method, size, step, jsonl_path) for every eval JSONL."""
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        jsonl = tag_dir / "drift_eval" / f"{tag_dir.name}.jsonl"
        if not jsonl.exists():
            continue
        info = parse_tag(tag_dir.name)
        if not info:
            continue
        method, size, step = info
        yield tag_dir.name, method, size, step, jsonl


def collect(root: Path):
    """Return nested dict: by[size][method][step][bench][metric] = value."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for tag, method, size, step, jsonl in discover(root):
        rows = parse_jsonl(jsonl)
        if not rows:
            continue
        _step, data = rows[-1]
        m = extract_metrics(data)
        for bench, vals in m.items():
            by[size][method][step][bench] = vals
    return by


# ----- formatters ---------------------------------------------------------

def fmt_text(by, benches_present):
    out = []
    for metric in ("mean@16", "best@16"):
        out.append(f"\n=== {metric} ===")
        header = ["size", "method", "step"] + [b for b in BENCH_ORDER if b in benches_present]
        widths = [max(len(h), 13) for h in header]
        out.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        out.append("  ".join("-" * w for w in widths))
        for size in sorted(by):
            for method in sorted(by[size], key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99):
                for step in sorted(by[size][method]):
                    row = [size, METHOD_DISPLAY.get(method, method), str(step)]
                    for b in [b for b in BENCH_ORDER if b in benches_present]:
                        v = by[size][method][step].get(b, {}).get(metric)
                        row.append(f"{v:.4f}" if v is not None else "  -   ")
                    out.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(out)


def fmt_markdown(by, benches_present):
    """One section per (size, metric): rows = (method, step), columns = benchmark."""
    benches = [b for b in BENCH_ORDER if b in benches_present]
    out = []
    out.append("<!-- Auto-generated by aggregate_eval.py — paste into docs/drift_method.md §6 -->")
    for size_key, size_label in (("1p7b", "Qwen3-1.7B-Base"), ("4b", "Qwen3-4B-Base")):
        if size_key not in by:
            continue
        for metric in ("mean@16", "best@16"):
            out.append("")
            out.append(f"### {size_label} — AIME/OOD {metric}")
            out.append("")
            header = "| Method | Step | " + " | ".join(benches) + " |"
            sep    = "|--------|------|" + "|".join(["------" for _ in benches]) + "|"
            out.append(header)
            out.append(sep)
            for method in sorted(by[size_key], key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99):
                disp = METHOD_DISPLAY.get(method, method)
                for step in sorted(by[size_key][method]):
                    cells = []
                    for b in benches:
                        v = by[size_key][method][step].get(b, {}).get(metric)
                        cells.append(f"{v:.4f}" if v is not None else "—")
                    out.append(f"| {disp} | {step} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def emit_csv(by, out_path: Path):
    rows = []
    for size in sorted(by):
        for method in sorted(by[size]):
            for step in sorted(by[size][method]):
                for bench, vals in by[size][method][step].items():
                    for metric, v in vals.items():
                        rows.append([size, METHOD_DISPLAY.get(method, method), step, bench, metric, v])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "method", "step", "benchmark", "metric", "value"])
        w.writerows(rows)


def emit_jsonl(by, out_path):
    """One JSON object per (method, size, step) with benchmark metrics flattened.

    Schema:
        {
          "method": "DRIFT", "method_run": "v18b", "size": "1p7b", "step": 120,
          "AIME24/mean@16": 0.039,  "AIME24/best@16":  0.177,
          "AIME25/mean@16": ...,
          "OlympiadBench/mean@16": ...,
          "GPQA/mean@16": ...,
        }
    """
    out_path = Path(out_path) if out_path is not None else None
    lines = []
    for size in sorted(by):
        for method in sorted(by[size], key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99):
            for step in sorted(by[size][method]):
                rec = {
                    "method": METHOD_DISPLAY.get(method, method),
                    "method_run": method,
                    "size": size,
                    "step": step,
                }
                for bench, vals in by[size][method][step].items():
                    for metric, v in vals.items():
                        rec[f"{bench}/{metric}"] = v
                lines.append(json.dumps(rec, sort_keys=True))
    out = "\n".join(lines) + ("\n" if lines else "")
    if out_path is not None:
        out_path.write_text(out)
    return out


# ----- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--format", choices=["text", "md", "csv", "jsonl"], default="text")
    ap.add_argument("--out", default=None, help="optional output path; default = stdout")
    ap.add_argument("--method", default=None, help="filter to a single method run (e.g. v18b, grpo, heg)")
    ap.add_argument("--size", default=None, help="filter to a single size (1p7b or 4b)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    by = collect(root)

    # Apply method/size filters if requested
    if args.method or args.size:
        filtered = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for size in by:
            if args.size and size != args.size:
                continue
            for method in by[size]:
                if args.method and method != args.method:
                    continue
                for step in by[size][method]:
                    for bench, vals in by[size][method][step].items():
                        filtered[size][method][step][bench] = vals
        by = filtered

    if not by:
        print("No eval results found (after filtering).", file=sys.stderr)
        sys.exit(1)

    benches_present = set()
    for size in by:
        for method in by[size]:
            for step in by[size][method]:
                benches_present.update(by[size][method][step].keys())

    if args.format == "csv":
        out_path = Path(args.out) if args.out else Path("eval_summary.csv")
        emit_csv(by, out_path)
        print(f"Wrote {out_path}", file=sys.stderr)
        return

    if args.format == "jsonl":
        text = emit_jsonl(by, args.out and Path(args.out))
        if args.out:
            print(f"Wrote {args.out}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return

    text = fmt_markdown(by, benches_present) if args.format == "md" else fmt_text(by, benches_present)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
