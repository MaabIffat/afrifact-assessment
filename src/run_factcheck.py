import argparse
import pandas as pd
from tqdm import tqdm

from openai_client import call_gpt4o
from prompts import (
    extensive_prompt,
    structured_prompt,
    simple_prompt,
    PROMPT_REGISTRY,  # <-- NEW
)
from language_config import get_language_instruction
from utils import parse_model_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--language", required=True)

    # UPDATED: add new prompt types to choices
    parser.add_argument(
        "--prompt_type",
        choices=[
            "extensive",
            "structured",
            "simple",
            "system_user_v1",
            "evidence_based_v2",
            "multilingual_v3",
        ],
        default="extensive",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Run only first N samples (for testing)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Optional: limit number of samples for testing
    if args.max_samples:
        df = df.head(args.max_samples)

    language_instruction = get_language_instruction(args.language)

    predictions = []
    justifications = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        claim = str(row.get("claim", ""))

        # Use your existing column name
        evidence = str(row.get("extracted_evidence_text", ""))

        # NEW: document field (if you have it), otherwise fallback to evidence
        document = str(row.get("displayed_text", evidence))

        # ---- NEW PROMPT TYPES (system+user) ----
        if args.prompt_type in PROMPT_REGISTRY:
            prompt_fn = PROMPT_REGISTRY[args.prompt_type]
            bundle = prompt_fn(
                claim=claim,
                evidence=evidence,
                document=document,
                language_instruction=language_instruction,
            )
            output = call_gpt4o(
                user_message=bundle.user_message,
                system_message=bundle.system_message,
            )

        # ---- OLD PROMPT TYPES (single user prompt) ----
        else:
            if args.prompt_type == "extensive":
                prompt = extensive_prompt(claim, evidence, language_instruction)
            elif args.prompt_type == "structured":
                prompt = structured_prompt(claim, evidence, language_instruction)
            else:
                prompt = simple_prompt(claim, evidence, language_instruction)

            # With the updated openai_client.py, pass as user_message
            output = call_gpt4o(user_message=prompt)

        label, justification = parse_model_output(output)
        predictions.append(label)
        justifications.append(justification)

    df["predicted_label"] = predictions
    df["justification"] = justifications

    df.to_csv(args.output_csv, index=False)
    print("Saved predictions to", args.output_csv)


if __name__ == "__main__":
    main()



# import argparse
# import pandas as pd
# from tqdm import tqdm
# from openai_client import call_gpt4o
# from prompts import extensive_prompt, structured_prompt, simple_prompt
# from language_config import get_language_instruction
# from utils import parse_model_output


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_csv", required=True)
#     parser.add_argument("--output_csv", required=True)
#     parser.add_argument("--language", required=True)
#     parser.add_argument("--prompt_type", choices=["extensive", "structured", "simple"], default="extensive")
#     parser.add_argument("--max_samples", type=int, default=None, help="Run only first N samples (for testing)")
#     args = parser.parse_args()

#     df = pd.read_csv(args.input_csv)
#     # Optional: limit number of samples for testing
#     if hasattr(args, "max_samples") and args.max_samples:
#     	df = df.head(args.max_samples)
#     language_instruction = get_language_instruction(args.language)

#     predictions = []
#     justifications = []

#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         claim = row["claim"]
#         evidence = row["extracted_evidence_text"]

#         if args.prompt_type == "extensive":
#             prompt = extensive_prompt(claim, evidence, language_instruction)
#         elif args.prompt_type == "structured":
#             prompt = structured_prompt(claim, evidence, language_instruction)
#         else:
#             prompt = simple_prompt(claim, evidence, language_instruction)

#         output = call_gpt4o(prompt)
#         label, justification = parse_model_output(output)

#         predictions.append(label)
#         justifications.append(justification)

#     df["predicted_label"] = predictions
#     df["justification"] = justifications

#     df.to_csv(args.output_csv, index=False)
#     print("Saved predictions to", args.output_csv)


# if __name__ == "__main__":
#     main()
