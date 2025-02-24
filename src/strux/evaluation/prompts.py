EVALUATOR_PAIRWISE_PREFERENCE_PROMPT = """
Evaluate and compare the {quality} of the two following outputs: 

Text A: {text_a}
Text B: {text_b}

Question: Which is more {quality}?
If text A is more {quality}, please return 'A'
If text B is more {quality}, please return 'B'
You must only return the choice.

Output in JSON format:
{{
    "preference": "A" or "B"
}}
"""
