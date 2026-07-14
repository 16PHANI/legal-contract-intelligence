def fill(template, contract_text):
    """Substitute contract text into prompt template safely.
    Uses <<<CONTRACT_TEXT>>> instead of .format() to avoid KeyError
    when legal text contains curly braces."""
    return template.replace("<<<CONTRACT_TEXT>>>", contract_text)


CLAUSE_EXTRACTION_PROMPT = """You are a senior legal analyst specialising in contract review.

Your task is to extract three specific clause types from the contract below.

--- STEP 1: REASONING (internal only, do not include in output) ---
Read the contract carefully. Identify sections that discuss:
  (a) When or how the agreement may be terminated.
  (b) Obligations around confidential or proprietary information.
  (c) Limits or caps on financial liability or indemnification.

--- STEP 2: OUTPUT ---
Return ONLY a valid JSON object with no markdown fences and no explanation.

JSON keys (all required):
  "termination_clause"      - verbatim text of the most comprehensive termination provision
  "confidentiality_clause"  - verbatim text of the primary confidentiality or NDA provision
  "liability_clause"        - verbatim text of the primary limitation-of-liability provision

If a clause type is entirely absent, set its value to the exact string "NOT_FOUND".
If multiple instances exist for one type, return the single most comprehensive passage.

--- EXAMPLE 1 ---
Contract excerpt:
"Either party may terminate this Agreement upon sixty (60) days prior written notice.
Each party shall maintain in confidence all Confidential Information of the other party
and shall not disclose it to any third party. Neither party shall be liable for any
indirect, incidental, or consequential damages."

Output:
{
  "termination_clause": "Either party may terminate this Agreement upon sixty (60) days prior written notice.",
  "confidentiality_clause": "Each party shall maintain in confidence all Confidential Information of the other party and shall not disclose it to any third party.",
  "liability_clause": "Neither party shall be liable for any indirect, incidental, or consequential damages."
}

--- EXAMPLE 2 ---
Contract excerpt:
"This Agreement shall remain in effect until terminated. Either party may terminate
immediately upon written notice if the other party materially breaches this Agreement
and fails to cure such breach within thirty (30) days of receiving written notice.
Vendor's total liability under this Agreement shall not exceed the fees paid by Client
in the twelve (12) months preceding the claim."

Output:
{
  "termination_clause": "Either party may terminate immediately upon written notice if the other party materially breaches this Agreement and fails to cure such breach within thirty (30) days of receiving written notice.",
  "confidentiality_clause": "NOT_FOUND",
  "liability_clause": "Vendor's total liability under this Agreement shall not exceed the fees paid by Client in the twelve (12) months preceding the claim."
}

--- CONTRACT ---
<<<CONTRACT_TEXT>>>

Return JSON with keys: termination_clause, confidentiality_clause, liability_clause"""


SUMMARY_PROMPT = """You are a senior legal analyst. Write a concise executive summary of the following legal contract.

Requirements:
  - Length: 100 to 150 words (count carefully).
  - Cover these three areas in order:
      1. Purpose: what the parties are agreeing to do.
      2. Obligations: the key duties each party must perform.
      3. Risks: notable penalties, liability caps, or breach consequences.
  - Tone: plain English, avoid unnecessary legal jargon.
  - Format: one continuous paragraph, no headers, no bullet points, no JSON.

Return ONLY the summary text.

--- CONTRACT ---
<<<CONTRACT_TEXT>>>"""
