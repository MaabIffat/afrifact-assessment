import argparse
import pandas as pd
from tqdm import tqdm
from openai_client import call_gpt4o
from prompts import extensive_prompt, structured_prompt, simple_prompt
from language_config import get_language_instruction
from utils import parse_model_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--prompt_type", choices=["extensive", "structured", "simple"], default="extensive")
    parser.add_argument("--max_samples", type=int, default=None, help="Run only first N samples (for testing)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    # Optional: limit number of samples for testing
    if hasattr(args, "max_samples") and args.max_samples:
    	df = df.head(args.max_samples)
    language_instruction = get_language_instruction(args.language)

    predictions = []
    justifications = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        claim = row["claim"]
        evidence = row["extracted_evidence_text"]

        if args.prompt_type == "extensive":
            prompt = extensive_prompt(claim, evidence, language_instruction)
        elif args.prompt_type == "structured":
            prompt = structured_prompt(claim, evidence, language_instruction)
        else:
            prompt = simple_prompt(claim, evidence, language_instruction)

        output = call_gpt4o(prompt)
        label, justification = parse_model_output(output)

        predictions.append(label)
        justifications.append(justification)

    df["predicted_label"] = predictions
    df["justification"] = justifications

    df.to_csv(args.output_csv, index=False)
    print("Saved predictions to", args.output_csv)


if __name__ == "__main__":
    main()
