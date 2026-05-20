# AGENTS Improvement Spec

Audit date: 2026-05-20  
Repository: `soufianeboulahcen/Master_Multimodal-Moroccan-SLG-main`  
Project: Multimodal Moroccan Sign Language Generation

---

## Audit summary

### What exists

| Artifact | State |
|---|---|
| `AGENTS.md` | Created today (was missing) |
| `.devcontainer/devcontainer.json` | Exists — generic universal image |
| Source code | None — zero commits |
| `.gitignore` | Missing |
| `requirements.txt` | Missing |
| `.ona/skills/` | Missing |
| `.cursor/rules/` | Missing |
| CI/CD config | Missing |
| Tests | Missing |

### What is good

- Dev container is present, so the environment is reproducible in principle.
- Repository is public and clearly named for its domain.
- `AGENTS.md` now exists with baseline guidance.

### What is missing

1. **`.gitignore`** — no protection against committing `__pycache__`, `*.pyc`, `venv/`, `data/`, model checkpoints, `.env` files.
2. **`requirements.txt` / `pyproject.toml`** — no dependency declaration; agents cannot install the project.
3. **Source code structure** — no `src/`, `tests/`, `configs/`, `notebooks/` directories.
4. **CI/CD** — no GitHub Actions workflow; no automated linting, testing, or type-checking.
5. **Agent skill files** — no `.ona/skills/` or `.cursor/rules/` to encode domain-specific workflows.
6. **Dev container specificity** — universal image is 10 GB and slow; no `postCreateCommand` to install deps.
7. **Dataset and checkpoint policy** — no documented storage strategy for large binary assets.
8. **Evaluation protocol** — no documented metrics or benchmark datasets for MSL generation.

### What is wrong

- The dev container has no `postCreateCommand`, so a fresh environment has no project dependencies installed.
- No `.gitignore` means the first `git add .` after code is written will likely stage `__pycache__`, virtual environments, or data files.
- `AGENTS.md` (newly created) describes a target layout that does not yet exist — it must be kept in sync as the project grows.

---

## Improvement spec

Each item below is a concrete, ordered action. Items are independent unless noted.

---

### 1. Add `.gitignore` immediately

**Priority:** critical — do before any code is written.

Create `.gitignore` covering:

```
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
venv/
.venv/
env/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data and models (store externally)
data/raw/
data/processed/
checkpoints/
*.pt
*.pth
*.ckpt
*.h5
*.pkl

# Environment
.env
.env.*
*.key

# IDE
.vscode/
.idea/
*.swp
.DS_Store
```

---

### 2. Replace universal dev container image

**Priority:** high — affects every environment start.

Replace `mcr.microsoft.com/devcontainers/universal:4.0.1-noble` with a Python-specific image and add dependency installation:

```jsonc
{
  "name": "Moroccan SLG",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install --upgrade pip && pip install -r requirements.txt",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-toolsai.jupyter"
      ]
    }
  }
}
```

Do this after `requirements.txt` exists (item 3).

---

### 3. Create `requirements.txt` (or `pyproject.toml`)

**Priority:** high — blocks environment setup and CI.

Minimum expected dependencies for an MSL generation project:

```
torch>=2.2
torchvision>=0.17
torchaudio>=2.2
transformers>=4.40
numpy>=1.26
scipy>=1.13
opencv-python-headless>=4.9
mediapipe>=0.10
librosa>=0.10
pyyaml>=6.0
tqdm>=4.66
pytest>=8.0
ruff>=0.4
mypy>=1.10
```

Adjust versions to match the actual model choices. Pin with `pip-compile` once stable.

---

### 4. Scaffold source layout

**Priority:** medium — needed before writing any model code.

Create the directory structure described in `AGENTS.md`:

```bash
mkdir -p src/{preprocessing,models,training,evaluation}
mkdir -p tests/{preprocessing,models,training,evaluation}
mkdir -p configs notebooks scripts data/{raw,processed}
touch src/__init__.py src/preprocessing/__init__.py src/models/__init__.py \
      src/training/__init__.py src/evaluation/__init__.py
```

Add a `data/.gitkeep` and ensure `data/raw/` and `data/processed/` are in `.gitignore`.

---

### 5. Add GitHub Actions CI

**Priority:** medium — enforces quality on every push.

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: ruff check src/ tests/
      - run: mypy src/
      - run: pytest tests/ -v --tb=short -m "not gpu"
```

GPU tests run only on self-hosted runners with `runs-on: [self-hosted, gpu]`.

---

### 6. Add domain-specific agent skill: `model-training`

**Priority:** medium — encodes the ML workflow so agents don't guess.

Create `.ona/skills/model-training.md`:

```markdown
# Skill: model-training

## When to use
When asked to train, fine-tune, or evaluate an MSL generation model.

## Workflow
1. Read `configs/<experiment>.yaml` for hyperparameters.
2. Verify dataset paths exist under `data/processed/`.
3. Run `python scripts/train.py --config configs/<experiment>.yaml`.
4. Log metrics to `runs/<experiment>/` (TensorBoard or W&B).
5. Save checkpoint to `checkpoints/<experiment>/best.pt`.
6. Run `python scripts/evaluate.py --checkpoint checkpoints/<experiment>/best.pt`.
7. Report BLEU-4, pose-distance, and FID scores.

## Anti-patterns
- Do not hardcode dataset paths in model files.
- Do not commit checkpoints to git.
- Do not run full training in a notebook.
```

---

### 7. Add domain-specific agent skill: `data-pipeline`

**Priority:** medium — MSL data handling is non-trivial.

Create `.ona/skills/data-pipeline.md`:

```markdown
# Skill: data-pipeline

## When to use
When asked to preprocess text, audio, or video for MSL generation.

## Workflow
1. Raw data lives in `data/raw/` — never modify it.
2. Run `python scripts/preprocess.py --input data/raw/ --output data/processed/`.
3. Validate output with `python scripts/validate_dataset.py`.
4. Document expected schema in `src/preprocessing/README.md`.

## Data format
- Input: Darija text (Arabic script) or audio (WAV, 16 kHz mono)
- Intermediate: gloss sequence (JSON)
- Output: pose sequence (NumPy `.npy`, shape `[T, J, 3]`)

## Anti-patterns
- Do not store raw video in git.
- Do not process the full dataset without first running on a 10-sample subset.
```

---

### 8. Expand `AGENTS.md` as code is added

**Priority:** ongoing.

After each major component is implemented, update `AGENTS.md` to reflect:

- Actual directory layout (replace "target" with "actual")
- Real commands for training, evaluation, and testing
- Any domain-specific gotchas discovered during development
- Dataset access instructions (without credentials)

---

## Prioritised action order

| # | Action | Effort | Blocks |
|---|---|---|---|
| 1 | Add `.gitignore` | 5 min | everything |
| 2 | Create `requirements.txt` | 30 min | 3, 5 |
| 3 | Update `devcontainer.json` | 10 min | — |
| 4 | Scaffold source layout | 15 min | 6, 7 |
| 5 | Add GitHub Actions CI | 20 min | — |
| 6 | Add `model-training` skill | 20 min | — |
| 7 | Add `data-pipeline` skill | 20 min | — |
| 8 | Keep `AGENTS.md` current | ongoing | — |
