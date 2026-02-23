# Deep Learning Research Repo Refactor & Hardening Plan

## Objective

This repository must be transformed into a **research-grade deep learning project** that is:

* Modular
* Reproducible
* Experiment-trackable
* Scalable
* Cleanly structured
* Publication-ready
* Robust to long-running experimentation

The current repository does **not** follow a proper research structure. The first goal is to reorganize the codebase into a standardized structure before any new features are added.

---

# Phase 1 — Restructure the Repository

## Target Structure

Refactor the repository into the following structure:

```
repo-name/
│
├── README.md
├── pyproject.toml (or requirements.txt if already used)
├── environment.yml (if using conda)
├── .gitignore
├── .pre-commit-config.yaml
│
├── configs/
│   ├── default.yaml
│   ├── model/
│   ├── data/
│   └── experiment/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   └── experiments/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── sweep.py
│
├── notebooks/
│
├── tests/
│
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
│
└── docs/
```

---

## Refactoring Rules

1. All model definitions must go inside:

   ```
   src/models/
   ```

2. All dataset and dataloading logic must go inside:

   ```
   src/data/
   ```

3. Training loops must go inside:

   ```
   src/training/
   ```

4. Evaluation logic must go inside:

   ```
   src/evaluation/
   ```

5. Utility functions must go inside:

   ```
   src/utils/
   ```

6. Entry-point scripts must go inside:

   ```
   scripts/
   ```

7. No training logic inside notebooks.

8. No hardcoded hyperparameters inside Python files — they must come from configs.

---

# Phase 2 — Configuration System

Implement a configuration system using YAML files.

## Requirements

* All hyperparameters must be configurable.
* No hardcoded values in training loops.
* Config must include:

  * model hyperparameters
  * training hyperparameters
  * seed
  * optimizer settings
  * logging settings

## Example Config Layout

```
configs/default.yaml
configs/model/transformer.yaml
configs/experiment/baseline.yaml
```

Example structure:

```yaml
model:
  latent_dim: 512
  hidden_dim: 768

training:
  batch_size: 64
  lr: 1e-4
  epochs: 100
  seed: 42

optimizer:
  type: adam
  weight_decay: 0.0
```

Training script must load config and pass values explicitly.

---

# Phase 3 — Reproducibility Requirements

The project must enforce reproducibility.

## Deterministic Setup

Training must set:

```python
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Logging Requirements

Each run must save:

```
results/checkpoints/<run_id>/
    model_last.pt
    model_best.pt
    config.yaml
    metrics.json
```

The exact config used must always be saved with the checkpoint.

---

# Phase 4 — Experiment Tracking Discipline

Create:

```
EXPERIMENTS.md
```

Format:

```
## Experiment 001
Date:
Goal:
Config:
Changes:
Results:
Observations:
Next Step:
```

Every experiment must be documented.

No undocumented experiments.

---

# Phase 5 — Required Markdown Files

Create the following files:

---

## 1. README.md

Must contain:

* Project description
* Research goal
* Setup instructions
* Minimal training command
* Repo structure explanation

---

## 2. ARCHITECTURE.md

Must describe:

* Model structure
* Mathematical formulation
* Loss components
* Design rationale
* Known limitations

---

## 3. EXPERIMENTS.md

Experiment log as described above.

---

## 4. REPRODUCIBILITY.md

Must include:

* Seed strategy
* Hardware used
* Software versions
* Deterministic settings
* Exact commands to reproduce main results

---

## 5. ROADMAP.md

High-level research plan:

Example:

```
Phase 1: Baseline implementation
Phase 2: Latent regularization
Phase 3: Dimensionality analysis
Phase 4: Scaling experiments
```

---

# Phase 6 — Logging & Monitoring

The training system must log:

* Total loss
* Individual loss components
* Learning rate
* Gradient norms
* Parameter count
* Model size
* Any research-specific metrics (e.g. latent KL per dimension)

Logging must support:

* TensorBoard or equivalent
* Saving metrics to disk

---

# Phase 7 — Testing

Create a `tests/` folder.

Add unit tests for:

* Model forward pass shape
* Loss outputs
* Deterministic behavior
* Config loading

Training must not silently break shapes.

---

# Phase 8 — Code Quality & Tooling

Add:

* black
* ruff
* mypy
* pre-commit

Create `.pre-commit-config.yaml`.

Add `.cursorignore`:

```
results/
checkpoints/
logs/
__pycache__/
```

---

# Phase 9 — Git Discipline

Branch strategy:

* main → stable
* dev → development
* feature/* → features
* experiment/* → experiment branches

Commit style:

```
feat: add KL per-dimension logging
fix: correct stride mismatch
exp: increase latent dimension to 768
refactor: move training loop to src/training
```

---

# Phase 10 — Notebook Policy

Notebooks may only be used for:

* Visualization
* Analysis
* Debugging

Never for core training logic.

---

# Final Instruction to Cursor

1. Analyze the current repository.
2. Move files into the new structure.
3. Refactor imports accordingly.
4. Implement configuration loading.
5. Remove hardcoded hyperparameters.
6. Create all required markdown files.
7. Ensure training still runs.
8. Add reproducibility controls.
9. Add logging and checkpoint saving.
10. Add basic tests.

The repository must be transformed into a clean, modular, research-grade deep learning project before further experimentation begins.


