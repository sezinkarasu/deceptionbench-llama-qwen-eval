import json, csv, os, glob

metric_dir = "."
models = ["llama3.1:8b", "qwen3:8b"]
domains = ["economy", "education", "entertainment", "healthcare", "social"]

def parse_filename(fname):
    stem = fname[:-len(".jsonl")]
    for model in models:
        if model in stem:
            idx = stem.index(model)
            before = stem[:idx].rstrip("_")
            after = stem[idx + len(model):].lstrip("_")
            tokens = [t for t in (before.split("_") if before else []) + (after.split("_") if after else []) if t]

            level, perspective, incentive, domain = "baseline", "", "", ""
            for t in tokens:
                if t in ("L1", "L2", "L3"):
                    level = t
                elif t in ("self", "other"):
                    perspective = t
                elif t in ("pressure", "reward"):
                    incentive = t
                elif t in domains:
                    domain = t
                elif t == "aggregated":
                    pass
                else:
                    print(f"  Warning: unrecognized token '{t}' in {fname}")

            if not domain:
                return None  # combined aggregated_metrics_<model>.jsonl file — skip
            return model, level, perspective, incentive, domain
    return None

rows = []

for fname in sorted(os.listdir(metric_dir)):
    if not fname.endswith(".jsonl"):
        continue
    parsed = parse_filename(fname)
    if parsed is None:
        continue
    model, level, perspective, incentive, domain = parsed

    path = os.path.join(metric_dir, fname)
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Skipping {fname}: not valid JSON ({e})")
            continue

    for condition, stats in data.items():
        for kind in ["thought", "response"]:
            if kind not in stats:
                continue
            honest = stats[kind]["honest"]
            decept = stats[kind]["decept"]
            total = honest + decept
            rate = round(decept / total, 3) if total else ""

            rows.append({
                "model": model,
                "level": level,
                "perspective": perspective,
                "incentive": incentive,
                "domain": domain,
                "condition": condition,
                "type": kind,
                "honest": honest,
                "decept": decept,
                "total": total,
                "deception_rate": rate,
            })

if not rows:
    print("No rows collected — check filenames/structure.")
else:
    out_path = "deception_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
