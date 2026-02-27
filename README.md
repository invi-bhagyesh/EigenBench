eigenbench\_

Packaged EigenBench evaluation framework.

## Modules

### `api.py`

LLM API wrappers. All models route through OpenRouter via `get_model_response`.

### `evaluation.py`

Framework for generating r\_{ijk} pairwise comparison samples.

**Functions:**

- `collect_responses` — get evaluee responses to a scenario
- `collect_reflections` — judge reflects on each response (multiturn only)
- `collect_comparison` — single pairwise comparison r\_{ijk}
- `evaluate_scenario` — full evaluation of one scenario
- `run_evaluation` — run over a dataset, saves JSONL incrementally

**Modes (via args):**

- `mode='criteria'` — per-criterion evaluation with `<criterion_N_choice>` tags (default)
- `mode='constitution'` — single constitution with `<choice>` tags
- `multiturn=True` — judge reflects on each response before comparing (default)
- `multiturn=False` — direct comparison, no reflection step
- `efficient=True` — track usage counts, prefer least-used judges/evaluees (inverse-weighted by alpha)

### `data_utils.py`

Extract structured comparison tuples from raw evaluation data and handle inconsistencies.

**Functions:**

- `extract_comparisons_with_ties` — for `<choice>` format → `[scenario, judge, eval1, eval2, score]`
- `extract_comparisons_with_ties_criteria` — for `<criterion>` format → `[criterion, scenario, judge, eval1, eval2, score]`
- `handle_inconsistencies_with_ties` — convert inconsistent pairs to ties (constitution)
- `handle_inconsistencies_with_ties_criteria` — same for criteria format
- `load_evaluations` — load from JSON or JSONL

### `bt.py`

Bradley-Terry / Bradley-Terry-Davidson model fitting.

**Models:**

- `VectorBT` — dot-product BT (binary)
- `VectorBT_norm` — euclidean-distance BT (binary)
- `VectorBT_bias` — dot-product BT with judge bias (binary)
- `VectorBTD` — dot-product BTD with ties (constitution, ternary)
- `VectorBTD_criteria` — BTD with separate judge embeddings per criterion (ternary)

**Training:**

- `train_vector_bt` — unified training loop (BCE for BT, CE for BTD)

### `eigentrust.py`

EigenTrust algorithm — compute trust scores from trained BT model.

**Functions:**

- `compute_trust_matrix` — trust matrix from BT model (no ties)
- `compute_trust_matrix_ties` — trust matrix from BTD model (with ties)
- `row_normalize`, `damp_matrix`, `eigentrust` — EigenTrust iteration
- `load_vector_bt`, `load_vector_btd` — load saved models

### `config.py`

Personas, constitutions, criteria. Copied from original. Requires JSON data files in `data/`.

## Tests

In `test/`:

- `test_evaluation.py` — run evaluation, save JSONL
- `test_data_bt.py` — extract comparisons from saved JSONL, fit BTD model
- `test_full_transcript.py` — data_utils + bt on full 37-model transcript
- `test_eigentrust.py` — full pipeline: data_utils → bt → eigentrust
