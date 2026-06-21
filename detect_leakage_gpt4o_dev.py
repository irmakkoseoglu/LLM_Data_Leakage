"""
Data Leakage Detection -- GPT-4o -- Zero to Six Shot Prompting
===============================================================
Kullanim:
    python detect_leakage_gpt4o.py --mode dev  --shots all   # tum shot sayilarini dene
    python detect_leakage_gpt4o.py --mode dev  --shots 3     # sadece 3-shot dene
    python detect_leakage_gpt4o.py --mode test --shots 3     # final test (bir kez!)

Gereksinimler:
    pip install openai pandas scikit-learn

API key:
    export OPENAI_API_KEY="sk-..."
"""

import os, time, argparse
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (precision_score, recall_score,
                             fbeta_score, f1_score, confusion_matrix)

# ── PATHS ─────────────────────────────────────────────────────────────────────
PIPELINES_DIR = "data/pipelines"
DEV_CSV       = "annotations/dev_set.csv"
TEST_CSV      = "annotations/test_set.csv"
MODEL         = "gpt-4o"
SLEEP         = 3.0   # saniye, rate limit icin

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM = """You are an expert machine learning engineer specializing in data leakage detection.

Data leakage in ML code occurs when information from the test set or future data is used during the training process, leading to overly optimistic performance estimates that do not generalize.

Common types of data leakage:
- preprocessing: scalers, imputers, or encoders fitted on the full dataset before train/test split
- overlap: training data contains rows from the test set (e.g. oversampling before split)
- multi-test: the same test set is used repeatedly for model selection or hyperparameter tuning
- temporal: time-series data split randomly instead of chronologically, allowing future data into training

Analyze the provided Python code and determine whether it contains data leakage.
Answer with exactly one word: yes or no."""

# ── FEW-SHOT EXAMPLES ─────────────────────────────────────────────────────────
EXAMPLES = {
    "preprocessing": {
        "code": """scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
model.fit(X_train, y_train)""",
        "label": "yes"
    },
    "multi-test": {
        "code": """X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier().fit(X_train, y_train)
gb = GradientBoostingClassifier().fit(X_train, y_train)
best = rf if accuracy_score(y_test, rf.predict(X_test)) > accuracy_score(y_test, gb.predict(X_test)) else gb""",
        "label": "yes"
    },
    "overlap": {
        "code": """X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
model.fit(X_train, y_train)""",
        "label": "yes"
    },
    "temporal": {
        "code": """df = df.sort_values('date')
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, shuffle=True)
model.fit(X_train, y_train)""",
        "label": "yes"
    },
    "none_1": {
        "code": """X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
model.fit(X_train_s, y_train)""",
        "label": "no"
    },
    "none_2": {
        "code": """pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))""",
        "label": "no"
    },
    "none_3": {
        "code": """tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    model.fit(X_tr, y[train_idx])
    print(model.score(X_te, y[test_idx]))""",
        "label": "no"
    }
}

# Shot sirasi: her shot yeni bir ornek ekler (0-shot = hic ornek yok)
SHOT_ORDER = [
    "preprocessing",
    "multi-test",
    "overlap",
    "temporal",
    "none_1",
    "none_2",
    "none_3"
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def build_messages(code: str, n_shots: int) -> list:
    messages = [{"role": "system", "content": SYSTEM}]
    for i in range(n_shots):
        ex = EXAMPLES[SHOT_ORDER[i]]
        messages.append({"role": "user",      "content": f"Does this code contain data leakage?\n\n{ex['code']}"})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": f"Does this code contain data leakage?\n\n{code}"})
    return messages


def read_code(filename: str) -> str:
    path = os.path.join(PIPELINES_DIR, filename)
    if not os.path.exists(path):
        return f"# File not found: {filename}"
    with open(path, "r", errors="ignore") as f:
        return f.read()


def detect(client: OpenAI, code: str, n_shots: int) -> int:
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=build_messages(code, n_shots),
                max_tokens=5,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().lower()
            return 1 if answer.startswith("yes") else 0
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  [retry {attempt+1}/5] Rate limit, waiting {wait}s...")
            time.sleep(wait)
    print("  [ERROR] Max retries reached, defaulting to 0")
    return 0


def evaluate(client: OpenAI, df: pd.DataFrame, n_shots: int, label: str = "") -> dict:
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        code = read_code(row["filename"])
        pred = detect(client, code, n_shots)
        true = 1 if str(row["has_leakage"]).strip().lower() in ("true", "1", "yes") else 0
        y_true.append(true)
        y_pred.append(pred)
        time.sleep(SLEEP)
        if len(y_true) % 5 == 0:
            print(f"  {len(y_true)}/{len(df)} processed...")

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tag = f" [{label}]" if label else ""
    print(f"\n{'='*50}")
    print(f"  {n_shots}-shot Results{tag}")
    print(f"{'='*50}")
    print(f"  Precision : {p:.3f}")
    print(f"  Recall    : {r:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  F2        : {f2:.3f}")
    print(f"  Confusion matrix:")
    print(f"  {cm}")
    print(f"{'='*50}\n")

    return {"shots": n_shots, "precision": p, "recall": r, "f1": f1, "f2": f2}


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["dev", "test"], default="dev",
                        help="dev: shot sayisini bul | test: final sonuc")
    parser.add_argument("--shots", default="all",
                        help="0-6 veya 'all'")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if args.mode == "dev":
        df = pd.read_csv(DEV_CSV)
        print(f"Development set: {len(df)} files")
        print(f"Model: {MODEL}\n")

        if args.shots == "all":
            results = []
            for n in range(7):   # 0, 1, 2, 3, 4, 5, 6
                print(f"\nTesting {n}-shot on development set...")
                r = evaluate(client, df, n, label="dev")
                results.append(r)

            # Ozet tablo
            print("\n\n" + "="*55)
            print("  SHOT COMPARISON SUMMARY (Development Set)")
            print("="*55)
            print(f"  {'Shots':<8} {'Precision':<12} {'Recall':<10} {'F1':<8} {'F2':<8}")
            print("  " + "-"*48)
            for r in results:
                print(f"  {r['shots']:<8} {r['precision']:<12.3f} {r['recall']:<10.3f} {r['f1']:<8.3f} {r['f2']:.3f}")
            print("="*55)

            best = max(results, key=lambda x: x["f2"])
            print(f"\n  Best F2 = {best['f2']:.3f}  at  {best['shots']}-shot")
            print(f"  >>> Run final test with: --mode test --shots {best['shots']}")

            os.makedirs("experiments", exist_ok=True)
            out_path = "experiments/gpt4o_dev_shot_results.csv"
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"\n  Results saved to {out_path}")

        else:
            n = int(args.shots)
            evaluate(client, df, n, label="dev")

    else:  # test
        if args.shots == "all":
            print("ERROR: --shots all sadece --mode dev ile kullanilabilir.")
            print("Once dev modunda en iyi shot sayisini bul, sonra test modunda calistir.")
            return

        df = pd.read_csv(TEST_CSV)
        n  = int(args.shots)

        print(f"FINAL TEST RUN")
        print(f"Model: {MODEL} | Strategy: {n}-shot | Files: {len(df)}")
        print("WARNING: Bu komutu sadece bir kez calistir!\n")

        result = evaluate(client, df, n, label="FINAL TEST")

        os.makedirs("experiments", exist_ok=True)
        out_path = f"experiments/gpt4o_final_test_{n}shot.csv"
        pd.DataFrame([result]).to_csv(out_path, index=False)
        print(f"Final results saved to {out_path}")


if __name__ == "__main__":
    main()
