# LLM-Based Detection of Data Leakage in Machine Learning Pipelines

> A comparative study of prompting strategies (zero-shot, few-shot, chain-of-thought) for detecting data leakage in Python ML pipelines using GPT-4o, benchmarked against Yang et al.'s static analysis baseline.

---

## Overview

Data leakage is a pervasive quality issue in machine learning pipelines — it occurs when information from the test set inadvertently influences model training, leading to inflated performance estimates that fail to generalize. Despite being well-documented, leakage is found in nearly **30% of public ML notebooks** (Yang et al., 2022).

Existing tools detect leakage through **static code analysis with hand-crafted rules**, which limits their ability to identify semantically complex patterns — particularly **temporal leakage** in time-series data. This project explores whether frontier LLMs can serve as generalizable leakage detectors without task-specific training or rules.

---

## Key Results

### Binary Detection (leakage present vs. absent)

| Method | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| Yang et al. (static analysis) | 0.824 | 0.808 | 0.816 | 0.811 |
| GPT-4o Zero-shot | 0.951 | 0.829 | 0.885 | 0.850 |
| GPT-4o Five-shot | **0.968** | 0.871 | 0.917 | 0.889 |
| GPT-4o CoT (optimized) | 0.954 | **0.886** | **0.919** | **0.899** |

### Per-Type F2 Scores

| Leakage Type | Yang et al. | Zero-shot | Five-shot | CoT |
|---|---|---|---|---|
| Preprocessing | 0.705 | 0.750 | 0.654 | 0.719 |
| Overlap | 0.327 | 0.747 | **0.805** | 0.795 |
| Multi-test | 0.679 | 0.471 | **0.677** | 0.667 |
| Temporal | — | **0.855** | 0.789 | 0.844 |
| None (clean) | — | 0.975 | **0.984** | **0.984** |

> **F2** weights recall 2× over precision — missing leakage (false negative) poses greater production risk than a false alarm.

---

## Research Questions

- **RQ1** — How does LLM-based detection compare to static analysis?
- **RQ2** — Which leakage types do LLMs detect most/least effectively?
- **RQ3** — Which prompting strategy yields the best performance?

---

## Repository Structure

```
LLM_Data_Leakage/
├── annotations/
│   ├── ground_truth.csv      # Full annotated dataset (117 pipelines)
│   ├── dev_set.csv           # Development set (20 pipelines, for prompt tuning)
│   └── test_set.csv          # Test set (96 pipelines, for final evaluation)
│
├── baselines/
│   └── leakage-analysis/     # Yang et al. (2022) static analysis tool
│
├── data/
│   └── pipelines/            # Python pipeline files (not tracked in git)
│
├── experiments/
│   ├── gpt4o_zeroshot_details.csv
│   ├── gpt4o_fewshot_details.csv
│   └── gpt4o_cot_details.csv
│
│
├── detect_leakage_gpt4o_dev.py   # Shot optimization on development set
└── detect_leakage_gpt4o_test.py  # Final evaluation on test set
```

---

## Dataset

The dataset consists of **96 unique Python ML pipelines** spanning five categories:

| Leakage Type | Description | Count |
|---|---|---|
| Preprocessing | Scaler/imputer/encoder fitted on full data before split | 26 |
| Overlap | Oversampling or augmentation applied before split | 18 |
| Multi-test | Same test set reused for model/hyperparameter selection | 19 |
| Temporal | Time-series data split randomly instead of chronologically | 15 |
| None | No leakage — correctly implemented pipeline | 18 |

**Sources:**
- **Real-world (39):** Sourced from Yang et al.'s benchmark (Kaggle and GitHub notebooks)
- **Collected (57):** Other pipelines are collected in order to create a balanced dataset.

> ⚠️ Pipeline `.py` files are not tracked in this repository due to licensing restrictions.

---

## Prompting Strategies

### Zero-shot
Model receives only a system prompt defining leakage types and returns structured JSON:
```json
{"has_leakage": true, "types": ["preprocessing"]}
```

### Five-shot
Five labeled examples (one per leakage type) are prepended as conversation turns before the query. Optimal shot count (5) was selected on the development set by evaluating 0–6 shots.

### Chain-of-Thought (CoT)
The model is instructed to analyze code in five sequential steps before concluding:
1. Locate the split boundary
2. Check for preprocessing leakage (any `.fit()` on unsplit data)
3. Check for overlap leakage (row-generating operations before split)
4. Check for multi-test leakage (test set used for decisions)
5. Check for temporal leakage (chronological order preserved?)

---

## Reproducing the Results

### Requirements

```bash
pip install openai pandas scikit-learn
export OPENAI_API_KEY="sk-..."
```

### Step 1 — Find optimal shot count (development set)

```bash
python detect_leakage_gpt4o_dev.py --mode dev --shots all
```

This evaluates 0–6 shots on the 20-file development set and reports F2 for each.

### Step 2 — Final evaluation (test set)

```bash
python detect_leakage_gpt4o_test.py --strategy fewshot   # 5-shot
python detect_leakage_gpt4o_test.py --strategy zeroshot  # 0-shot
python detect_leakage_gpt4o_test.py --strategy cot       # chain-of-thought
```

Results are saved to `experiments/` with per-file predictions and per-type breakdowns.

---

## References

1. Yang, C., Brower-Sinning, R. A., Lewis, G., & Kastner, C. (2022). *Data leakage in notebooks: Static detection and better processes.* ASE '22. [[Paper]](https://arxiv.org/abs/2209.03345) [[Tool]](https://github.com/malusamayo/leakage-analysis)

2. Alturayeif, N., & Hassine, J. (2025). *Data leakage detection in machine learning code: Transfer learning, active learning, or low-shot prompting?* PeerJ Computer Science, 11, e2730. [[Paper]](https://peerj.com/articles/cs-2730/)

3. AlOmar, E. A., DeMario, C., Shagawat, R., & Kreiser, B. (2025). *LeakageDetector: An open source data leakage analysis tool in machine learning pipelines.* arXiv:2503.14723.

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{koseoglu2025llm,
  title     = {LLM-Based Detection of Data Leakage in Machine Learning Pipelines: 
               A Comparative Study of Prompting Strategies},
  author    = {Koseoglu, Irmak},
  year      = {2025},
  note      = {Manuscript in preparation}
}
```

---

## License

Code is released under the MIT License. Dataset annotations are released under CC BY 4.0.
