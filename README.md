# Synthetic Diagnostic Reasoning -- Visible CoT + DeepSeek-R1 LoRA Fine-Tuning

This repository provides a complete pipeline for visible
Chain-of-Thought (CoT) fine-tuning, including:

-   formal problem domain (`data/domain_rules.json`)
-   deterministic solver (`src/solver.py`)
-   reproducible CoT dataset generator (`src/generate_dataset.py`)
-   optional teacher CoT generation via Camel/OpenAI
    (`src/generate_cot_with_camel.py`)
-   LoRA / QLoRA fine-tuning (`src/finetune_lora.py`)
-   evaluation (`src/evaluate.py`)
-   inference CLI (`src/infer.py`)

------------------------------------------------------------------------

# 1. Installation

``` bash
pip install -r requirements.txt
```

Optional dependency for teacher-based CoT generation (Unstable!):

``` bash
pip install -r requirements-camel.txt
```

------------------------------------------------------------------------

# 2. Generate CoT Dataset Locally (Deterministic) (I would recommend this! This is more stable!)

``` bash
python src/generate_dataset.py --seed 42 --n_samples 12000 --unknown_ratio 0.2 --out_dir data
```

This creates:

-   `data/train.jsonl`
-   `data/val.jsonl`
-   `data/test.jsonl`

Dataset schema (per line):

-   `id`
-   `symptoms`
-   `instruction`
-   `input`
-   `output` (visible CoT + final line `Final answer: <LABEL>`)
-   `label`

------------------------------------------------------------------------

# 3. Optional: Generate CoT with a Teacher Model (Camel/OpenAI) (Unstable!)

Example using the OpenAI backend:

``` bash
python src/generate_cot_with_camel.py \
  --input_file data/train.jsonl \
  --output_file data/train_teacher_cot.jsonl \
  --provider openai \
  --model_name gpt-4o \
  --max_samples 1000
```

The generator discards samples when the produced **Final answer** does
not match the gold label.

------------------------------------------------------------------------

# 4. Start Fine-Tuning

``` bash
CUDA_VISIBLE_DEVICES=0 python src/finetune_lora.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --train_file data/train.jsonl \
  --val_file data/val.jsonl \
  --output_dir outputs/adapters \
  --run_id ds_r1_diag_lora \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --max_length 256 \
  --lr 2e-4 \
  --qlora
```

Note: `src/finetune_lora.py` supports **Alpaca-style CoT datasets**
(`instruction` / `input` / `output`) while remaining backward compatible
with legacy `input` / `output` formats.

------------------------------------------------------------------------

# 5. Run Evaluation

``` bash
python src/evaluate.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --test_file data/test.jsonl \
  --save_report outputs/report.json
```

Evaluation is **label-centric** (Accuracy / Exact Match).\
Label extraction prioritizes:

1.  `Final answer: <LABEL>`
2.  Regex fallback

------------------------------------------------------------------------

# 6. Example Inference

Default output prints **visible reasoning + final answer**:

``` bash
python src/infer.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --symptoms "fever,cough,fatigue,headache,myalgia"
```

Optional compact output:

``` bash
python src/infer.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --symptoms "fever,cough,fatigue,headache,myalgia" \
  --output label
```

------------------------------------------------------------------------


# Domain Definition

`data/domain_rules.json` contains:

-   68 symptoms
-   15 diseases (`Disease_01` to `Disease_15`)

Each disease defines:

-   `required`
-   `optional`
-   `exclude`

Tie-break rule:

-   choose the candidate with the **highest score**
-   if tied → choose the **lexicographically smallest disease ID**

Unknown rule:

-   if no diagnosis satisfies all required symptoms and zero excluded
    symptoms → return `UNKNOWN`

Scoring function:

    score = 5 * required_matches + optional_matches - 0.2 * extra_symptoms

------------------------------------------------------------------------

# Notes

-   Labels are determined by the deterministic solver.
-   Reproducibility is ensured via a random seed.
-   A GPU with sufficient VRAM is recommended for fine-tuning.
