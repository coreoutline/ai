# Experiments

This folder contains training scripts and exploratory notebooks for the CoreOutline transformer project. All Python training scripts import from `src/` at the repository root, so **run commands from the repository root** unless noted otherwise.

```
transformer/
├── data/                  # Local datasets and checkpoints inputs
├── models/                # Saved model weights (.pth)
├── src/                   # Core library (models, training, inference)
└── experiments/
    ├── training/          # Runnable training scripts
    └── notebooks/         # Jupyter notebooks for exploration
```

## Prerequisites

1. **Python 3.10+** (3.11 recommended for the fine-tuning scripts).

2. **Install dependencies** from the repo root:

```bash
pip install -r requirements.txt
```

3. **Additional packages** depending on which experiment you run:

| Experiment | Extra packages |
|---|---|
| Baseline Ray Tune (`train.py`) | `ray[tune]` |
| Qwen pretraining (`train_qwen.py`) | `wandb` |
| Instruction fine-tuning (DDP) | CUDA-capable PyTorch with distributed support |
| Hugging Face datasets / uploads | `huggingface_hub`, `pyarrow` |
| Notebooks (varies) | `jupyter`, `matplotlib`, `nltk`, etc. |

4. **GPU**: Recommended for all training scripts. CPU will work for smoke tests but will be slow.

5. **Hugging Face**: Several scripts and notebooks load data or tokenizers from the Hub (`Qwen/Qwen1.5-0.5B`, `DeividasM/financial-instruction-aq22`, etc.). Ensure you are logged in if accessing private repos:

```bash
huggingface-cli login
```

---

## Training scripts

All commands below assume your shell is at the **repository root** (`transformer/`).

### 1. Baseline transformer — Ray Tune hyperparameter search

**Script:** `experiments/training/train.py`

Trains the baseline `CoreModel` (GPT-style decoder) on text data using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) with an ASHA scheduler. Configuration comes from `src/config/baseline_config.py`.

**Data:** Expects `data/verdict.txt` one level above the repo root (the script resolves `../../../data/verdict.txt` from `experiments/training/`). Place the file at `<parent-of-transformer>/data/verdict.txt`, or edit the path in the script to point at `./data/verdict.txt` inside this repo.

```bash
pip install "ray[tune]"
python experiments/training/train.py
```

---

### 2. Small Qwen model — code corpus pretraining

**Script:** `experiments/training/train_qwen.py`

Trains a lightweight CoreOutline Qwen architecture on a slice of `code_contents.txt`. Logs metrics to Weights & Biases (`project="chaos"`).

**Data:** Opens `../../../data/code_contents.txt` relative to the **current working directory**. Run from `experiments/training/` so it resolves to `<parent-of-transformer>/data/code_contents.txt`, or change the path in the script to `./data/code_contents.txt` and run from the repo root.

```bash
pip install wandb
cd experiments/training
python train_qwen.py
```

---

### 3. Instruction fine-tuning (single GPU)

**Script:** `experiments/training/fine_tune_instruct.py`

Fine-tunes the full CoreOutline Qwen model on financial instruction data from Hugging Face. Loads base weights from `./models/nyx_2.pth` if present.

**Data:** Downloaded automatically from `hf://datasets/DeividasM/financial-instruction-aq22/...`

```bash
python experiments/training/fine_tune_instruct.py
```

Default training: 2 epochs, batch size 8.

---

### 4. Instruction fine-tuning (multi-GPU / DDP)

**Scripts:**
- `experiments/training/fine_tune_instruct_ddp.py` — standard DDP training
- `experiments/training/fine_tune_instruct_ddp_optimized.py` — optimized variant (same entry point pattern)
- `experiments/training/launch_training.py` — convenience wrapper that detects GPUs and invokes `torchrun`

**Requires:** CUDA and multiple GPUs for meaningful speedup (works on one GPU with `--nproc_per_node=1`).

From repo root:

```bash
cd experiments/training
torchrun --nproc_per_node=<NUM_GPUS> fine_tune_instruct_ddp.py
```

Or use the launcher (auto-detects GPU count):

```bash
cd experiments/training
python launch_training.py
```

Base checkpoint: `./models/nyx_2.pth` (relative to repo root when the script runs).

---

### 5. Financial reasoning fine-tuning

**Script:** `experiments/training/fine_tune_reasoning.py`

Fine-tunes CoreOutline Qwen on chain-of-thought financial Q&A. Data loading and prompt formatting live in `src/training/reasoning_data.py`.

**Data:** CSV with columns `prompt`, `context`, `thinking`, and `answer` (default: `./data/finetuning_llm.csv`).

**Smoke test:**

```bash
python experiments/training/fine_tune_reasoning.py \
  --csv-path ./data/finetuning_llm.csv \
  --checkpoint ./models/nyx_2_reasoning.pth \
  --epochs 1 \
  --batch-size 2 \
  --allowed-max-length 2048
```

**Full run (example):**

```bash
python experiments/training/fine_tune_reasoning.py \
  --csv-path ./data/finetuning_llm.csv \
  --checkpoint ./models/nyx_2_reasoning.pth \
  --epochs 100 \
  --batch-size 4 \
  --allowed-max-length 4096
```

| Flag | Default | Description |
|---|---|---|
| `--csv-path` | `./data/finetuning_llm.csv` | Training CSV |
| `--checkpoint` | `./models/nyx_2_reasoning.pth` | Load/save weights |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `4` | Batch size |
| `--eval-freq` | `5` | Validate every N steps |
| `--eval-iter` | `2` | Validation batches per eval |
| `--allowed-max-length` | `4096` | Max sequence length |
| `--num-workers` | `0` | DataLoader workers |

---

### 6. Tool-calling fine-tuning (xLAM)

**Script:** `experiments/training/fine_tune_tool_calling.py`

Fine-tunes CoreOutline Qwen on Salesforce xLAM function-calling examples. Data loading lives in `src/training/tool_calling_data.py`.

**Data:** `./data/xlam_function_calling_60k.csv` with columns `query`, `answers` (JSON), and `tools` (JSON). You can also explore or export this dataset in `notebooks/Untitled2.ipynb` or `notebooks/salesforce_tools_.ipynb`.

**Smoke test (1k samples):**

```bash
python3 experiments/training/fine_tune_tool_calling.py \
  --csv-path ./data/xlam-function-calling-60k.csv \
  --checkpoint ./models/nyx_2_latest_1.pth \
  --base-checkpoint ./models/nyx_2_tool_calling.pth \
  --epochs 1 \
  --batch-size 4 \
  --max-samples 1000
```

**Full run (example):**

```bash
python experiments/training/fine_tune_tool_calling.py \
  --csv-path ./data/xlam_function_calling_60k.csv \
  --checkpoint ./models/nyx_2_tool_calling.pth \
  --base-checkpoint ./models/nyx_2.pth \
  --epochs 3 \
  --batch-size 4 \
  --lr 5e-5
```

| Flag | Default | Description |
|---|---|---|
| `--csv-path` | `./data/xlam_function_calling_60k.csv` | xLAM CSV |
| `--checkpoint` | `./models/nyx_2_tool_calling.pth` | Fine-tuned output weights |
| `--base-checkpoint` | `./models/nyx_2.pth` | Optional base model to initialize from |
| `--max-samples` | `None` (full dataset) | Limit rows for quick tests |
| `--lr` | `5e-5` | Learning rate |
| Other flags | Same as reasoning script | See table above |

---

### Utility: `refactor.py`

One-off maintenance script used to refactor training files by slicing and inserting shared imports from `utils.py`. **Not an experiment** — do not run unless you are actively refactoring those scripts.

---

## Notebooks

Start Jupyter from the repo root so relative paths resolve correctly:

```bash
jupyter lab
# or
jupyter notebook
```

Open notebooks under `experiments/notebooks/`. Set the kernel working directory to the repository root if imports fail.

### Core model & training exploration

| Notebook | Purpose |
|---|---|
| `Pytorch2_Transformer_Text_Generation.ipynb` | End-to-end decoder-only transformer walkthrough: data loading, tokenization, training loop |
| `CoreGPT.ipynb` | Blog-data tokenization experiments (SentencePiece, Hugging Face datasets) |
| `GemmaFineTuning.ipynb` | Fine-tuning and Hub upload workflow for Gemma 2B IT |

### Data preparation & Hub

| Notebook | Purpose |
|---|---|
| `dataset_upload_hf.ipynb` | Upload local dataset folders to Hugging Face Hub (`huggingface_hub.upload_folder`) |
| `Extract Prompts.ipynb` | Prompt extraction utilities |
| `Untitled2.ipynb` / `salesforce_tools_.ipynb` | Explore Salesforce xLAM function-calling dataset |

### NLP utilities & downstream tasks

| Notebook | Purpose |
|---|---|
| `question_answering.ipynb` | Hugging Face `question-answering` pipeline demos (including PDF context) |
| `context_retrieval.ipynb` | Wikipedia-based context gathering for RAG-style prompts |
| `topic_modelling.ipynb` | Topic modelling scratch notebook (mostly empty — template) |
| `sentence_word_tokenization.ipynb` | NLTK stopwords and stemming examples |

### Ad-hoc / archived notebooks

The `notebooks/ad-hoc-notebooks/` folder holds one-off experiments (text classification with TensorFlow/Keras, transfer learning, model distillation, PaddleNLP, etc.). These often reference local paths on specific machines (e.g. `D://CoreOutline/data/...`) — update paths before running.

---

## Data & checkpoints checklist

| Asset | Used by | Location |
|---|---|---|
| `verdict.txt` | `train.py` | `<parent-of-transformer>/data/verdict.txt` (or edit script to use `./data/`) |
| `code_contents.txt` | `train_qwen.py` | Same parent `data/` folder when run from `experiments/training/` |
| `finetuning_llm.csv` | `fine_tune_reasoning.py` | `data/finetuning_llm.csv` |
| `xlam_function_calling_60k.csv` | `fine_tune_tool_calling.py` | `data/xlam_function_calling_60k.csv` |
| `nyx_2.pth` | Instruction & tool-calling fine-tunes | `models/nyx_2.pth` |
| Financial instruction parquet | `fine_tune_instruct*.py` | Hugging Face Hub (auto-download) |

Create the `models/` directory if it does not exist; training scripts save checkpoints there by default.

---

## Troubleshooting

- **`ModuleNotFoundError: src`** — Run from the repository root, or ensure `sys.path` inserts in the script point to the repo root (they should by default).
- **CUDA OOM** — Lower `--batch-size` or `--allowed-max-length`; enable gradient checkpointing (already on in fine-tuning scripts).
- **Missing data file** — See the checklist above; several notebooks can download or export datasets from Hugging Face.
- **DDP hangs** — Confirm `torchrun --nproc_per_node` matches available GPUs; try `--nproc_per_node=1` to isolate issues.
- **Ray Tune errors** — Install Ray: `pip install "ray[tune]"`.
