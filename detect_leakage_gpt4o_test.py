"""
Data Leakage Detection -- GPT-4o -- Multiple Prompting Strategies
==================================================================
Kullanim:
    python detect_leakage_final_gpt4o.py --strategy zeroshot
    python detect_leakage_final_gpt4o.py --strategy fewshot
    python detect_leakage_final_gpt4o.py --strategy cot
    python detect_leakage_final_gpt4o.py --strategy all

Gereksinimler:
    pip install openai pandas scikit-learn
    export OPENAI_API_KEY="sk-..."
"""

import os, time, json, argparse
import pandas as pd
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

PIPELINES_DIR = "data/pipelines"
TEST_CSV      = "annotations/test_set.csv"
MODEL         = "gpt-4o"
SLEEP         = 3.0

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────
SYSTEM_BASE = """You are an expert machine learning engineer specializing in data leakage detection.

Data leakage occurs when information from the test set or future data is used during training, leading to overly optimistic performance that does not generalize.

Types of data leakage:
- preprocessing: scalers, imputers, or encoders fitted on full dataset before train/test split
- overlap: training data contains rows from test set (e.g. oversampling before split)
- multi-test: same test set reused repeatedly for model selection or hyperparameter tuning
- temporal: time-series data split randomly instead of chronologically

Respond ONLY with a JSON object in this exact format:
{"has_leakage": true/false, "types": ["type1", "type2"]}

If no leakage: {"has_leakage": false, "types": []}
Types must be from: preprocessing, overlap, multi-test, temporal"""

SYSTEM_COT = """You are an expert machine learning engineer specializing in data leakage detection.

Data leakage occurs when information from the test set or future data influences the training process, leading to inflated performance metrics that do not generalize to unseen data. Note that these leakage types are not mutually exclusive — a single pipeline may contain multiple types simultaneously.

Analyze the provided Python code using the following steps:

Step 1 — Locate the split boundary.
Find where the data is divided into training and test sets. Look for train_test_split(), iloc slicing, manual index splitting, or time-based cutoffs. If no explicit split is found, treat the entire pipeline as potentially contaminated and continue checking for other leakage signals.

Step 2 — Check for preprocessing leakage.
Look for any operation that LEARNS from the unsplit dataset before the split boundary. The key signal is a .fit() or .fit_transform() call on data that has not yet been split — regardless of what the transformer is. Also look for statistics computed on the unsplit dataset (mean, std, min, max, vocabulary, principal components, correlation values) that are then used to transform both training and test data. If any information from future test rows influences a transformation applied during training, this is preprocessing leakage.

Step 3 — Check for overlap leakage.
Look for any operation that generates new rows or duplicates existing ones — such as oversampling, undersampling with replacement, data augmentation, or synthesis — applied to the unsplit dataset before the split. If rows derived from test data end up in the training set, this is overlap leakage.

Step 4 — Check for multi-test leakage.
Check whether the test set is used to make any decision beyond a single final evaluation. Even if the test set is not explicitly passed to a training function, if its performance score is used to select between models, configurations, features, or thresholds, this constitutes multi-test leakage. The test set should be touched exactly once.

Step 5 — Check for temporal leakage.
If the data contains time-related columns (dates, timestamps, sequential IDs, or any ordering variable), verify that the split respects chronological order. A random shuffle of time-series data before splitting allows future observations to leak into the training set.

Step 6 — Conclude.
Based on your analysis above, output ONLY the JSON object below on the last line of your response. Do not include your step-by-step reasoning in the final output — only the JSON:
{"has_leakage": true/false, "types": ["type1", "type2"]}

Rules:
- types must only contain values from: preprocessing, overlap, multi-test, temporal
- if multiple leakage types are present, list all of them
- if no leakage is found: {"has_leakage": false, "types": []}"""

# ── FEW-SHOT EXAMPLES (5-shot) ────────────────────────────────────────────────
EXAMPLES = [
    {
        "code": "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)\nmodel.fit(X_train, y_train)",
        "label": '{"has_leakage": true, "types": ["preprocessing"]}'
    },
    {
        "code": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nrf = RandomForestClassifier().fit(X_train, y_train)\ngb = GradientBoostingClassifier().fit(X_train, y_train)\nbest = rf if accuracy_score(y_test, rf.predict(X_test)) > accuracy_score(y_test, gb.predict(X_test)) else gb",
        "label": '{"has_leakage": true, "types": ["multi-test"]}'
    },
    {
        "code": "X_res, y_res = SMOTE().fit_resample(X, y)\nX_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)\nmodel.fit(X_train, y_train)",
        "label": '{"has_leakage": true, "types": ["overlap"]}'
    },
    {
        "code": "df = df.sort_values('date')\nX_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1), df['target'], test_size=0.2, shuffle=True)\nmodel.fit(X_train, y_train)",
        "label": '{"has_leakage": true, "types": ["temporal"]}'
    },
    {
        "code": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s = scaler.transform(X_test)\nmodel.fit(X_train_s, y_train)",
        "label": '{"has_leakage": false, "types": []}'
    }
]


# ── MESSAGE BUILDERS ──────────────────────────────────────────────────────────
def build_zeroshot(code, system):
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Analyze this code:\n\n{code}"}
    ]

def build_fewshot(code, system):
    msgs = [{"role": "system", "content": system}]
    for ex in EXAMPLES:
        msgs.append({"role": "user",      "content": f"Analyze this code:\n\n{ex['code']}"})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": f"Analyze this code:\n\n{code}"})
    return msgs

def build_cot(code, system):
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Analyze this code step by step:\n\n{code}"}
    ]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def read_code(filename):
    path = os.path.join(PIPELINES_DIR, filename)
    if not os.path.exists(path):
        return f"# File not found: {filename}"
    with open(path, "r", errors="ignore") as f:
        return f.read()


def detect(client, code, strategy):
    if strategy == "zeroshot":
        msgs = build_zeroshot(code, SYSTEM_BASE)
        max_tokens = 60
    elif strategy == "fewshot":
        msgs = build_fewshot(code, SYSTEM_BASE)
        max_tokens = 60
    else:  # cot
        msgs = build_cot(code, SYSTEM_COT)
        max_tokens = 300  # CoT icin daha uzun

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()

            # JSON bul — CoT durumunda ekstra metin olabilir
            if "{" in raw:
                json_str = raw[raw.rfind("{"):raw.rfind("}")+1]
                result = json.loads(json_str)
            else:
                result = {"has_leakage": False, "types": []}

            has_leakage = bool(result.get("has_leakage", False))
            types = [t.lower().strip() for t in result.get("types", [])]
            return has_leakage, types

        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  [retry {attempt+1}/5] {e} — waiting {wait}s...")
            time.sleep(wait)

    return False, []


def evaluate(client, gt, strategy):
    print(f"\nRunning strategy: {strategy.upper()}")
    print(f"Files: {len(gt)}\n")

    rows = []
    for i, row in gt.iterrows():
        code = read_code(row["filename"])
        pred_has, pred_types = detect(client, code, strategy)
        rows.append({
            "filename":     row["filename"],
            "true_leakage": row["has_leakage"],
            "pred_leakage": pred_has,
            "true_types":   str(row["true_types"]),
            "pred_types":   str(pred_types),
        })
        time.sleep(SLEEP)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(gt)} processed...")

    results = pd.DataFrame(rows)

    # Genel metrikler
    y_true = results["true_leakage"].astype(int).tolist()
    y_pred = results["pred_leakage"].astype(int).tolist()

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    print(f"\n{'='*55}")
    print(f"  GPT-4o [{strategy}] -- RESULTS")
    print(f"{'='*55}")
    print(f"  Precision : {p:.3f}")
    print(f"  Recall    : {r:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  F2        : {f2:.3f}")

    # Per-type analiz
    print(f"\n  Per-type Analysis:")
    print(f"  {'Type':<20} {'TP':>4} {'FN':>4} {'FP':>4} {'Recall':>8} {'Precision':>10} {'F1':>6} {'F2':>6}")
    print(f"  {'-'*62}")

    type_results = {}
    for lt in ["preprocessing", "overlap", "multi-test", "temporal", "none"]:
        tp = fn = fp = 0
        for _, row in results.iterrows():
            true_types = eval(row["true_types"]) if row["true_types"] not in ("[]","nan") else []
            pred_types = eval(row["pred_types"]) if row["pred_types"] not in ("[]","nan") else []

            if lt == "none":
                if not row["true_leakage"]:
                    if not row["pred_leakage"]: tp += 1
                    else: fp += 1
            else:
                in_true = lt in true_types
                in_pred = lt in pred_types
                if in_true and in_pred:       tp += 1
                elif in_true and not in_pred: fn += 1
                elif not in_true and in_pred: fp += 1

        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1t  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f2t  = (1 + 4) * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0
        print(f"  {lt:<20} {tp:>4} {fn:>4} {fp:>4} {rec:>8.3f} {prec:>10.3f} {f1t:>6.3f} {f2t:>6.3f}")
        type_results[lt] = {"tp": tp, "fn": fn, "fp": fp, "recall": rec, "precision": prec, "f1": f1t, "f2": f2t}

    print(f"{'='*55}\n")

    # Kaydet
    os.makedirs("experiments", exist_ok=True)
    results.to_csv(f"experiments/gpt4o_{strategy}_details.csv", index=False)
    print(f"  Saved to experiments/gpt4o_{strategy}_details.csv")

    return {
        "strategy": strategy,
        "precision": p, "recall": r, "f1": f1, "f2": f2,
        "per_type": type_results
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["zeroshot", "fewshot", "cot", "all"],
                        default="all")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    df = pd.read_csv(TEST_CSV)

    # Ground truth: dosya bazinda tekillestirilmis
    gt = df.groupby("filename").agg(
        has_leakage=("has_leakage", lambda x: any(
            str(v).lower() in ("true","1","yes") for v in x)),
        true_types=("leakage_type", lambda x: list(set([
            t for t in x if str(t).lower() not in ("none","null","nan","")])))
    ).reset_index()

    print(f"Test set: {len(gt)} unique files")

    strategies = ["zeroshot", "fewshot", "cot"] if args.strategy == "all" else [args.strategy]

    all_results = []
    for s in strategies:
        r = evaluate(client, gt, s)
        all_results.append(r)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Strategy':<12} {'Precision':>10} {'Recall':>8} {'F1':>6} {'F2':>6}")
        print(f"  {'-'*44}")
        for r in all_results:
            print(f"  {r['strategy']:<12} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>6.3f} {r['f2']:>6.3f}")
        print(f"{'='*60}")

        # Ozet kaydet
        summary = pd.DataFrame([{k: v for k, v in r.items() if k != "per_type"}
                                  for r in all_results])
        summary.to_csv("experiments/gpt4o_strategy_comparison.csv", index=False)
        print(f"\n  Summary saved to experiments/gpt4o_strategy_comparison.csv")


if __name__ == "__main__":
    main()
