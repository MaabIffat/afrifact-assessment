def extensive_prompt(claim, evidence, language_instruction):
    return f"""
{language_instruction}

You are a senior scientific fact-checking expert.

TASK:
Determine whether the claim is:
- SUPPORTS
- REFUTES
- NOT_ENOUGH_INFORMATION

CRITERIA:
1. Only use the provided evidence.
2. Do NOT use external knowledge.
3. Check:
   - Numerical consistency
   - Temporal alignment
   - Geographic consistency
   - Logical implications
   - Contradictions
4. If evidence partially supports but lacks completeness → NOT_ENOUGH_INFORMATION.
5. If direct contradiction → REFUTES.
6. If clearly entailed → SUPPORTS.

INPUT:
Claim:
\"\"\"{claim}\"\"\"

Evidence:
\"\"\"{evidence}\"\"\"

OUTPUT FORMAT:
Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
Justification: <Detailed reasoning explaining step-by-step evaluation>
"""

def structured_prompt(claim, evidence, language_instruction):
    return f"""
{language_instruction}

Fact-check the claim using ONLY the provided evidence.

Steps:
1. Identify key factual components in claim.
2. Locate matching or conflicting information in evidence.
3. Compare them.
4. Decide label.

Claim:
{claim}

Evidence:
{evidence}

Return:
Predicted_Label: <SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION>
Justification: <Concise explanation>
"""

def simple_prompt(claim, evidence, language_instruction):
    return f"""
{language_instruction}

Does the evidence support the claim?

Claim:
{claim}

Evidence:
{evidence}

Answer strictly in this format:
Predicted_Label: SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION
Justification: <short explanation>
"""
