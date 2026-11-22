# Week 3 — MLE-Ops: Training curves and real prediction handoff

These additions are **new files only**. They do not alter Anas's `src/train.py`,
`src/core_model.py`, or Lionel's API. They do not change `main` until the PR is reviewed.

## 1. Install new dependencies (from repo root on Windows)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-week3-extras.txt
```

## 2. Run genuine training and save charts (no checkpoint overwritten in Git)

```powershell
.\.venv\Scripts\python.exe -m scripts.week3_curves --run --protocol loo --c 4 --seed 0
```

This creates **observed** files in `reports/week3/`:
- `training_loo_c4_seed0.log` (complete stdout of the new run)
- `training_curves.csv` (training retriever loss and validation metrics)
- `retriever_training_loss.png`
- `validation_ndcg.png` (validation NDCG@10 for retriever and ranker)

The retriever script prints a *sum of batch losses per epoch*, not normalized
mean loss. The existing ranker script **does not print ranker training loss**;
do not invent a missing ranker loss curve. To protect Anas's tracked checkpoint,
new weights are saved under `%USERPROFILE%\Downloads\AgentQuiver_Week3_training\loo_c4_seed0\`.

## 3. Real Top-K predictions from Ahmed's **full-model** checkpoint

Use the full checkpoint you saved after the full-model experiment (not the
No Text or No Age checkpoint):

```powershell
.\.venv\Scripts\python.exe -m scripts.week3_predict --username 88subarno88 --k 5 --checkpoint "$HOME\Downloads\ckpt_loo_c4_trained_week3.pt"
```

The username must exist in `data/processed/users.csv`. You can choose others
from that table. Predictions use all interactions available up to `as_of`
(default: current UTC time); historical target holdouts are **not** preserved
for this serving-style demo. Scores are uncalibrated logits/similarity,
**not probabilities**. Items are npm/PyPI *packages*, not GitHub repositories.

The class interface for Lionel is:

```python
from scripts.week3_predict import ModelPredictor
predictor = ModelPredictor("path/to/full_model_checkpoint.pt")
rows = predictor.predict("88subarno88", k=5)
```

### Important integration issue for Lionel

`src/api/recommender.py` currently uses `MockRecommender`, whose API schema
represents GitHub **repositories** with owner/repo IDs. This trained model ranks
npm/PyPI **packages**. The API contract and item-to-repository mapping need an
explicit team decision. Do not label these packages as GitHub repos or pretend
raw scores are calibrated probabilities. No API integration or live deploy is
claimed by these files.

### Scope and caveats

- The model checkpoint's metadata currently does not record `no_age`; use the
  separately saved **full-model** checkpoint for this serving example.
- Dataset item metadata is loaded from the provided full table. Leakage policy
  for this transductive setup still needs agreement before claiming Q3 as Yes.
- Chart code parses newly captured, actual `src.train` output; no synthetic
  reports are substituted.
- The local PR is not the same as a merged PR; request teammate review.

## 4. Test, add, and commit to the **existing** feature branch

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git status --short
git add scripts/week3_predict.py scripts/week3_curves.py tests/test_week3_predict_curves.py requirements-week3-extras.txt reports/week3/HANDOFF_MLE_OPS.md
# After generating curves, add the observed log/csv/png evidence deliberately.
git commit -m "feat: add Week 3 real predictions and training-curve capture"
git push
```
