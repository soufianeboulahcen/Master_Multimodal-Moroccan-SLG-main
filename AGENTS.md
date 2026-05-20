# AGENTS.md

Agent guidance for the **Multimodal Moroccan Sign Language Generation** project.

---

## Project overview

This repository targets automatic generation of Moroccan Sign Language (MSL/LSM) from text or speech input using multimodal deep-learning techniques. The expected stack is Python-based (PyTorch or TensorFlow), with data pipelines, model training scripts, and evaluation tooling.

> **Status:** repository initialised — no source code committed yet.  
> Update this file as the codebase grows.

---

## Repository layout (target)

```
.
├── data/               # Raw and processed datasets (not committed — see .gitignore)
├── src/
│   ├── preprocessing/  # Text/audio → gloss pipeline
│   ├── models/         # Architecture definitions
│   ├── training/       # Training loops and configs
│   └── evaluation/     # Metrics (BLEU, WER, pose-distance, etc.)
├── notebooks/          # Exploratory analysis only — no production logic
├── tests/              # Unit and integration tests
├── configs/            # YAML/TOML experiment configs
├── scripts/            # One-off CLI helpers
├── requirements.txt    # Pinned dependencies
└── AGENTS.md           # This file
```

---

## Environment setup

The dev container uses `mcr.microsoft.com/devcontainers/universal:4.0.1-noble`.  
Once source code is added, replace it with a leaner Python image and pin dependencies:

```jsonc
// .devcontainer/devcontainer.json
{
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install -r requirements.txt"
}
```

---

## Coding conventions

- **Language:** Python 3.10+
- **Style:** PEP 8, enforced by `ruff` (line length 100)
- **Type hints:** required on all public functions and class methods
- **Docstrings:** Google style
- **Tests:** `pytest`; place under `tests/` mirroring `src/` structure
- **Notebooks:** exploratory only; strip outputs before committing

---

## Agent rules

### General

- Read the file before editing it.
- Do not commit data files, model checkpoints, or generated outputs.
- Do not hardcode paths — use `pathlib.Path` and config files.
- Do not expose API keys or dataset credentials in code or logs.

### When adding a new model

1. Define the architecture in `src/models/`.
2. Add a config schema in `configs/`.
3. Register the model in `src/models/__init__.py`.
4. Add at least one unit test in `tests/models/`.

### When modifying data pipelines

1. Verify the pipeline against a small sample before running on the full dataset.
2. Document expected input/output formats in the module docstring.
3. Add a smoke test that runs in < 5 seconds without GPU.

### Commit messages

Follow Conventional Commits:

```
<type>(<scope>): <short description>

Types: feat | fix | refactor | test | docs | chore | perf
```

Example: `feat(models): add transformer-based pose decoder`

### Pull requests

- Target `main` unless working on a feature branch.
- PR title must follow the same Conventional Commits format.
- Include: motivation, what changed, how to test.
- Do not merge without passing CI.

---

## Testing

```bash
pytest tests/ -v --tb=short
```

For GPU-dependent tests, mark with `@pytest.mark.gpu` and skip in CI unless a GPU runner is available.

---

## Known constraints

- MSL dataset is not publicly available; agents must not attempt to download or generate synthetic data without explicit instruction.
- Model checkpoints can be large (>1 GB); store them in a separate artifact store, not in git.
