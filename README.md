# Multilingual Scientific Fact-Checking using GPT-4o

This repository performs multilingual scientific fact-checking using GPT-4o.

## Supported Languages
- Amharic
- English
- Hausa
- Swahili
- Yoruba
- Zulu

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY in .env


RUN

```bash
python src/run_factcheck.py \
  --input_csv data.csv \
  --output_csv predictions.csv \
  --language english \
  --prompt_type extensive \
  --max_samples 5

Output

Original CSV columns are preserved.

New columns added:
	•	predicted_label
	•	justification

python
>>> from src.evaluator import evaluate_predictions
>>> evaluate_predictions("predictions.csv")

Labels
	•	SUPPORTS
	•	REFUTES
	•	NOT_ENOUGH_INFORMATION

This design supports:

- Prompt ablation studies
- Multilingual experiments
- Cross-language consistency evaluation
- Justification quality analysis
- Zero-shot fact-checking

This can be extended to 10+ African languages by adding them in `language_config.py`.


