import re


def parse_model_output(output_text):
    label_match = re.search(r"Predicted_Label:\s*(SUPPORTS|REFUTES|NOT_ENOUGH_INFORMATION)", output_text, re.IGNORECASE)
    justification_match = re.search(r"Justification:\s*(.*)", output_text, re.DOTALL)

    label = label_match.group(1).upper() if label_match else "PARSE_ERROR"
    justification = justification_match.group(1).strip() if justification_match else output_text.strip()

    return label, justification
