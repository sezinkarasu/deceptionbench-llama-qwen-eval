import json, os, csv
 
RESULT_DIR = os.path.expanduser("~/DeceptionBench/result")
GEN_DIR = os.path.join(RESULT_DIR, "generation")
EVAL_DIR = os.path.join(RESULT_DIR, "eval")
OUT_PATH = os.path.join(RESULT_DIR, "metric", "full_responses.csv")
 
models = ["llama3.1:8b", "qwen3:8b"]
domains = ["economy", "education", "entertainment", "healthcare", "social"]
 
 
def parse_condition_from_filename(fname):
    """Extract model/level/perspective/incentive/domain from a filename like
    output_L2_self_pressure_llama3.1:8b_economy.jsonl or
    eval_L1_other_qwen3:8b_social.jsonl
    """
    stem = fname
    for prefix in ("output_", "eval_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    stem = stem[: -len(".jsonl")]
 
    for model in models:
        if model in stem:
            idx = stem.index(model)
            before = stem[:idx].rstrip("_")
            after = stem[idx + len(model):].lstrip("_")
            tokens = [t for t in before.split("_") if t] + [t for t in after.split("_") if t]
 
            level = perspective = incentive = domain = ""
            for t in tokens:
                if t in ("L1", "L2", "L3"):
                    level = t
                elif t in ("self", "other"):
                    perspective = t
                elif t in ("pressure", "reward"):
                    incentive = t
                elif t in domains:
                    domain = t
            return model, level, perspective, incentive, domain
    return None
 
 
def expected_condition_key(level, perspective, incentive):
    """Reconstruct the hyphenated key used inside the JSON, e.g. 'L2-self-pressure'."""
    parts = [level, perspective]
    if incentive:
        parts.append(incentive)
    return "-".join(p for p in parts if p)
 
 
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: bad JSON at {path}:{line_num} ({e})")
    return records
 
 
rows = []
 
for turn_type in ["single_turn", "multi_turn"]:
    gen_dir = os.path.join(GEN_DIR, turn_type)
    eval_dir = os.path.join(EVAL_DIR, turn_type)
    if not os.path.isdir(gen_dir) or not os.path.isdir(eval_dir):
        print(f"Skipping {turn_type}: folder missing")
        continue
 
    for fname in sorted(os.listdir(gen_dir)):
        if not (fname.startswith("output_") and fname.endswith(".jsonl")):
            continue
 
        parsed = parse_condition_from_filename(fname)
        if not parsed:
            print(f"  Warning: could not parse {fname}, skipping")
            continue
        model, level, perspective, incentive, domain = parsed
 
        eval_fname = "eval_" + fname[len("output_"):]
        gen_path = os.path.join(gen_dir, fname)
        eval_path = os.path.join(eval_dir, eval_fname)
 
        if not os.path.exists(eval_path):
            print(f"  Warning: no matching eval file for {fname} (expected {eval_fname})")
            continue
 
        gen_records = load_jsonl(gen_path)
        eval_records = load_jsonl(eval_path)
 
        # index generation records by question text for joining
        gen_by_question = {r.get("question", ""): r for r in gen_records}
 
        fallback_key = expected_condition_key(level, perspective, incentive)
 
        for erec in eval_records:
            question = erec.get("question", "")
            gen_rec = gen_by_question.get(question)
 
            eval_block = erec.get("eval", {})
            # normal shape: {"L1-self": {"thought": ..., "response": ...}}
            # seen malformed shape: {"thought": ..., "response": ...} (no condition wrapper)
            if "thought" in eval_block and "response" in eval_block:
                verdict_map = {fallback_key: eval_block}
            else:
                verdict_map = eval_block
 
            for cond_key, verdict in verdict_map.items():
                response_text = ""
                if gen_rec:
                    response_text = gen_rec.get("responses", {}).get(cond_key, "")
 
                rows.append({
                    "model": model,
                    "turn_type": turn_type,
                    "level": level,
                    "perspective": perspective,
                    "incentive": incentive,
                    "domain": domain,
                    "condition_key": cond_key,
                    "topic": erec.get("topic", ""),
                    "dimension": erec.get("dimension", ""),
                    "question": question,
                    "goal": erec.get("goal", ""),
                    "groundtruth": erec.get("groundtruth", ""),
                    "thought_verdict": verdict.get("thought", ""),
                    "response_verdict": verdict.get("response", ""),
                    "response_text": response_text,
                    "timestamp": erec.get("timestamp", ""),
                })
 
if not rows:
    print("No rows collected — check folder paths and filenames.")
else:
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
 

