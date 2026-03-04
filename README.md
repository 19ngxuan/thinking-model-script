# Synthetic Diagnostic Reasoning - Dataset + DeepSeek-R1 LoRA Fine-Tuning

Dieses Repository liefert ein komplettes Abgabe-Paket:
- formale Problemdomäne (`data/domain_rules.json`)
- deterministischer Solver (`src/solver.py`)
- reproduzierbarer Datensatzgenerator (`src/generate_dataset.py`)
- LoRA/QLoRA Fine-Tuning (`src/finetune_lora.py`)
- Evaluation (`src/evaluate.py`)
- Inference-CLI (`src/infer.py`)
- beigefügter Datensatz (`data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`)

## 1) Installation

```bash
pip install -r requirements.txt
```

## 2) Dataset generieren (mit Seed)

```bash
python src/generate_dataset.py --seed 42 --n_samples 6000 --unknown_ratio 0.2 --out_dir data
```

Erzeugt:
- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`

## 3) Fine-Tuning starten

```bash
python src/finetune_lora.py \
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

Output:
- `outputs/adapters/ds_r1_diag_lora/`

## 4) Evaluation ausführen

```bash
python src/evaluate.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --test_file data/test.jsonl \
  --save_report outputs/report.json
```

Konsolen-Output:
- Accuracy / Exact Match

Datei:
- `outputs/report.json`

## 5) Beispiel-Inference

```bash
python src/infer.py \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter_path outputs/adapters/ds_r1_diag_lora \
  --symptoms "fever,cough,fatigue,headache,myalgia"
```

## 6) Tests ausführen

```bash
python -m unittest discover -s tests -v
```

## Domain-Definition

`data/domain_rules.json` enthält:
- 68 Symptome
- 15 Krankheiten (`Disease_01` bis `Disease_15`)
- pro Krankheit: `required`, `optional`, `exclude`
- Tiebreak-Regel: höchster Score, bei Gleichstand lexikografisch kleinste Disease-ID
- Unknown-Regel: keine Diagnose gültig => `UNKNOWN`

Score:
- `score = 5 * required_matches + optional_matches - 0.2 * extra_symptoms`

## Hinweise

- Labels werden im Generator immer über den deterministischen Solver erzeugt.
- Reproduzierbarkeit wird durch fixed seed sichergestellt.
- Für Fine-Tuning wird eine GPU mit ausreichendem VRAM empfohlen.
