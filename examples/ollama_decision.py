"""Minimal local HTTP example. Requires Ollama >=0.35 with nimble pulled.

Uses only Python's standard library. It does not execute the selected parser.
"""

import json
from urllib.request import Request, urlopen


def evaluate(state, *, endpoint="http://127.0.0.1:11434/v1/systemone"):
    criteria = {
        "invoice_csv_v1": "Invoice identifier, supplier, amount and due date",
        "inventory_csv_v1": "Product identifier, quantity and stock location",
        "review": "Insufficient evidence or neither supported format",
    }
    payload = {
        "model": "nimble",
        "state": state,
        "keep_alive": "10m",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which supported parser fits the evidence?",
                "criteria": criteria,
            }
        },
    }
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    if len(body) > 65536:
        raise ValueError("Request exceeds Ollama's 64 KiB body limit")
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    answer = result["answers"]["route"]
    if answer.get("type") != "choice" or answer.get("choice") not in criteria:
        raise ValueError("Invalid decision returned by backend")
    # Return native metadata unchanged. Acceptance thresholds need evaluation.
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate({
        "header": ["Inv No", "Supplier", "Amount", "Due Date"],
        "sample_rows": [["A42", "Example Supplier", "125.00", "2026-09-30"]],
    }), indent=2))
