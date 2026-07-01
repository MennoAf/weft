"""Personal-Agent Acceptance Harness (PAAH).

Proves Weft *answers* real personal-agent query shapes correctly, with
STRUCTURAL assertions over seeded synthetic data — not LLM-judged, not
LongMemEval. See the design spec in Weft memory ``weft-b9582b8f``.

This first slice covers the **enumeration / counting** shape end-to-end:
seed a known-cardinality corpus through the real ``weft_remember`` write path,
then ask ``weft_recall`` "how many plants do I have" through the real read
path and assert on *what an agent would actually answer* — proving the
enumeration answer block (explicit ``count`` + complete ``members`` list, V8)
closes the consumption contract that a naive ``results[]`` read left open.
"""
