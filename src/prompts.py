# src/prompts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable


@dataclass(frozen=True)
class PromptBundle:
    """
    A prompt bundle that can provide:
      - system_message: goes into the system role
      - user_message: goes into the user role
    """
    system_message: str
    user_message: str


def _label_note() -> str:
    # We keep canonical labels exactly as your pipeline expects:
    # SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION
    return """
IMPORTANT LABELS (must be exact, case-sensitive):
- SUPPORTS
- REFUTES
- NOT_ENOUGH_INFORMATION
"""


def prompt_system_user_v1(claim: str, evidence: str, document: str, language_instruction: str) -> PromptBundle:
    """
    Your first new prompt: SYSTEM/USER style with JSON-like keys.
    (We keep output format stable for parsing: Predicted_Label + Justification.)
    """
    system_msg = (
        "You are a helpful assistant for automated fact-checking. "
        "Your task is to analyze claims based on provided evidence only."
    )

    user_msg = f"""
{language_instruction}

You are an intelligent decision support system designed for automated fact-checking.
Based only on the evidence provided below, determine whether the evidence SUPPORTS,
REFUTES, or provides NOT_ENOUGH_INFORMATION for the given claim.

Expected fields (for your reasoning):
"Veracity": "SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION"
"Justification": "Detailed reasoning addressing clarity, relevance, consistency, and sufficiency of the evidence"

Definitions for Veracity labels:
- SUPPORTS: The claim is accurate and there’s nothing significant missing.
- REFUTES: The claim is inaccurate, contradicted by the evidence, or makes an incorrect assertion.
- NOT_ENOUGH_INFORMATION: The evidence is insufficient, unrelated, or does not provide enough relevant information.

{_label_note()}

Document:
\"\"\"{document}\"\"\"

Evidence:
\"\"\"{evidence}\"\"\"

Claim:
\"\"\"{claim}\"\"\"

Question:
Based only on the evidence above, does the evidence SUPPORT, REFUTE, or provide NOT_ENOUGH_INFORMATION for the claim?

OUTPUT FORMAT (must follow exactly):
Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
Justification: <Detailed reasoning>
""".strip()

    return PromptBundle(system_message=system_msg, user_message=user_msg)


def evidence_based_fact_checking_v2(claim: str, evidence: str, document: str, language_instruction: str) -> PromptBundle:
    """
    Your second new prompt: Evidence-Based Fact Checking (role/objective/constraints).
    """
    system_msg = "You are a professional fact checker."

    user_msg = f"""
{language_instruction}

Evidence-Based Fact Checking

Role:
You are a professional fact checker responsible for verifying factual claims.

Objective:
Assess the relationship between a claim and the provided evidence by determining whether
the evidence supports the claim, contradicts it, or does not provide enough information.

Constraints:
- Use only the evidence provided below.
- Do not rely on prior knowledge, assumptions, or external information.
- If the evidence does not clearly support or contradict the claim, select NOT_ENOUGH_INFORMATION.

{_label_note()}

Document:
\"\"\"{document}\"\"\"

Evidence:
\"\"\"{evidence}\"\"\"

Claim:
\"\"\"{claim}\"\"\"

Question:
Based only on the evidence above, does the evidence SUPPORT, REFUTE,
or provide NOT_ENOUGH_INFORMATION for the claim?

OUTPUT FORMAT (must follow exactly):
Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
Justification: <Detailed reasoning>
""".strip()

    return PromptBundle(system_message=system_msg, user_message=user_msg)


def multilingual_evidence_centered_v3(claim: str, evidence: str, document: str, language_instruction: str) -> PromptBundle:
    """
    Your third new prompt: Multilingual Evidence-Centered Fact Verification.
    """
    system_msg = (
        "You are a helpful assistant for Multilingual Evidence-Centered Fact Verification. "
        "Your task is to analyze claims based on evidence, possibly across languages."
    )

    user_msg = f"""
{language_instruction}

You are an intelligent decision support system designed for automated fact-checking.
Based only on the evidence provided below, determine whether the evidence SUPPORTS,
REFUTES, or provides NOT_ENOUGH_INFORMATION for the given claim.

Role:
You are an independent fact checker tasked with evaluating factual claims in linguistically diverse settings.

Objective:
Determine the factual status of a claim by interpreting the provided evidence, which may appear in different languages
or linguistic varieties, and deciding whether the evidence confirms the claim, contradicts it, or fails to address it.

Constraints:
- Base your judgment exclusively on the evidence provided, irrespective of language or linguistic variation.
- Do not use background knowledge, assumptions, or external sources.
- If the evidence does not directly confirm or contradict the claim, choose NOT_ENOUGH_INFORMATION.

{_label_note()}

Document:
\"\"\"{document}\"\"\"

Evidence:
\"\"\"{evidence}\"\"\"

Claim:
\"\"\"{claim}\"\"\"

Question:
Based only on the evidence above, does the evidence SUPPORT, REFUTE,
or provide NOT_ENOUGH_INFORMATION for the claim?

OUTPUT FORMAT (must follow exactly):
Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
Justification: <Detailed reasoning>
""".strip()

    return PromptBundle(system_message=system_msg, user_message=user_msg)


# -------------------------------------------------------------------
# Backwards-compatible wrappers (if your pipeline still calls these)
# -------------------------------------------------------------------
def extensive_prompt(claim: str, evidence: str, language_instruction: str) -> str:
    """
    Backwards compatibility: returns a single combined user prompt.
    Uses v2 (evidence-based) by default.
    """
    bundle = evidence_based_fact_checking_v2(
        claim=claim,
        evidence=evidence,
        document=evidence,  # fallback
        language_instruction=language_instruction,
    )
    # If the caller only supports a single prompt string, we return the user message.
    return bundle.user_message


def structured_prompt(claim: str, evidence: str, language_instruction: str) -> str:
    """
    Backwards compatibility: use v1.
    """
    bundle = prompt_system_user_v1(
        claim=claim,
        evidence=evidence,
        document=evidence,
        language_instruction=language_instruction,
    )
    return bundle.user_message


def simple_prompt(claim: str, evidence: str, language_instruction: str) -> str:
    """
    Backwards compatibility: simple version.
    """
    system_msg = "You are a fact-checking assistant."
    user_msg = f"""
{language_instruction}

Decide the label using ONLY the evidence.

{_label_note()}

Claim:
\"\"\"{claim}\"\"\"

Evidence:
\"\"\"{evidence}\"\"\"

OUTPUT FORMAT (must follow exactly):
Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
Justification: <1-2 sentences>
""".strip()
    # single-string compatibility
    return user_msg


# -------------------------------------------------------------------
# Prompt registry for run_factcheck.py
# -------------------------------------------------------------------
PROMPT_REGISTRY: Dict[str, Callable[[str, str, str, str], PromptBundle]] = {
    # New prompt types (recommended)
    "system_user_v1": prompt_system_user_v1,
    "evidence_based_v2": evidence_based_fact_checking_v2,
    "multilingual_v3": multilingual_evidence_centered_v3,
}


# def extensive_prompt(claim, evidence, language_instruction):
#     return f"""
# {language_instruction}

# You are a senior scientific fact-checking expert.

# TASK:
# Determine whether the claim is:
# - SUPPORTS
# - REFUTES
# - NOT_ENOUGH_INFORMATION

# CRITERIA:
# 1. Only use the provided evidence.
# 2. Do NOT use external knowledge.
# 3. Check:
#    - Numerical consistency
#    - Temporal alignment
#    - Geographic consistency
#    - Logical implications
#    - Contradictions
# 4. If evidence partially supports but lacks completeness → NOT_ENOUGH_INFORMATION.
# 5. If direct contradiction → REFUTES.
# 6. If clearly entailed → SUPPORTS.

# INPUT:
# Claim:
# \"\"\"{claim}\"\"\"

# Evidence:
# \"\"\"{evidence}\"\"\"

# OUTPUT FORMAT:
# Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
# Justification: <Detailed reasoning explaining step-by-step evaluation>
# """

# def structured_prompt(claim, evidence, language_instruction):
#     return f"""
# {language_instruction}

# Fact-check the claim using ONLY the provided evidence.

# Steps:
# 1. Identify key factual components in claim.
# 2. Locate matching or conflicting information in evidence.
# 3. Compare them.
# 4. Decide label.

# Claim:
# {claim}

# Evidence:
# {evidence}

# Return:
# Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
# Justification: <Concise explanation>
# """

# def simple_prompt(claim, evidence, language_instruction):
#     return f"""
# {language_instruction}

# Does the evidence support the claim?

# Claim:
# {claim}

# Evidence:
# {evidence}

# Answer strictly in this format:
# Predicted_Label: SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION
# Justification: <short explanation>
# """
