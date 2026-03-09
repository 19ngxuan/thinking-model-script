# Synthetic Diagnostic Reasoning - Visible CoT + DeepSeek-R1 LoRA Fine-Tuning

Dieses Repository liefert ein komplettes Paket fuer sichtbares Chain-of-Thought (CoT) Fine-Tuning:
- formale Problemdomaene (`data/domain_rules.json`)
- deterministischer Solver (`src/solver.py`)
- reproduzierbarer CoT-Datensatzgenerator (`src/generate_dataset.py`)
- optionaler Teacher-CoT-Generator via Camel/OpenAI (`src/generate_cot_with_camel.py`)
- LoRA/QLoRA Fine-Tuning (`src/finetune_lora.py`)
- Evaluation (`src/evaluate.py`)
- Inference-CLI (`src/infer.py`)

## 1) Installation

```bash
pip install -r requirements.txt
```

Optional fuer Teacher-basierte CoT-Generierung:

```bash
pip install -r requirements-camel.txt
```

## 2) CoT-Dataset lokal generieren (deterministisch)

```bash
python src/generate_dataset.py --seed 42 --n_samples 12000 --unknown_ratio 0.2 --out_dir data
```

Erzeugt:
- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`

Datensatzschema (pro Zeile):
- `id`
- `symptoms`
- `instruction`
- `input`
- `output` (sichtbares CoT + finale Zeile `Final answer: <LABEL>`)
- `label`

## 3) Optional: CoT mit Teacher-Modell (Camel/OpenAI) erzeugen

Beispiel mit OpenAI-Backend:

```bash
python src/generate_cot_with_camel.py \
  --input_file data/train.jsonl \
  --output_file data/train_teacher_cot.jsonl \
  --provider openai \
  --model_name gpt-4o \
  --max_samples 1000
```

Der Generator verwirft Samples, wenn `Final answer` nicht dem Gold-Label entspricht.

## 4) Fine-Tuning starten

```bash
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

Hinweis: `src/finetune_lora.py` unterstuetzt jetzt Alpaca-CoT (`instruction`/`input`/`output`) und bleibt rueckwaertskompatibel zu Legacy-`input`/`output`.

## 5) Evaluation ausfuehren

```bash
python src/evaluate.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --test_file data/test.jsonl \
  --save_report outputs/report.json
```

Evaluation bleibt label-zentriert (Accuracy/Exact Match), Label-Extraktion priorisiert:
- `Final answer: <LABEL>`
- Regex-Fallback

## 6) Beispiel-Inference

Standardausgabe ist jetzt sichtbares CoT + finale Antwort:

```bash
python src/infer.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --symptoms "fever,cough,fatigue,headache,myalgia"
```

Optionale kompakte Ausgabe:

```bash
python src/infer.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --symptoms "fever,cough,fatigue,headache,myalgia" \
  --output label
```

## 7) Tests ausfuehren

```bash
python -m unittest discover -s tests -v
```

## Domain-Definition

`data/domain_rules.json` enthaelt:
- 68 Symptome
- 15 Krankheiten (`Disease_01` bis `Disease_15`)
- pro Krankheit: `required`, `optional`, `exclude`
- Tiebreak-Regel: hoechster Score, bei Gleichstand lexikografisch kleinste Disease-ID
- Unknown-Regel: keine Diagnose gueltig => `UNKNOWN`

Score:
- `score = 5 * required_matches + optional_matches - 0.2 * extra_symptoms`

## Hinweise

- Labels werden weiterhin durch den deterministischen Solver festgelegt.
- Reproduzierbarkeit wird durch Seed sichergestellt.
- Fuer Fine-Tuning wird eine GPU mit ausreichendem VRAM empfohlen.
