"""Use Ollama's real decision endpoint. Never substitute a heuristic move."""
import json
import math
import random
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

POLICY = (
    "Choose the Tetris placement that best keeps the game alive and clears rows. "
    "Avoid creating buried holes: they are very difficult to repair. Prefer more completed rows "
    "among placements with similarly few holes, then a low, compact, smooth stack. "
    "Compare the numeric outcomes in each option. All options are legal. "
    "There is no hold. Return the placement to actually execute."
)


def describe(move):
    f = move.metrics
    return (f"Rotate {move.rotation} degrees, left column {move.x + 1}; "
            f"rows cleared {move.cleared}; resulting buried holes {f['holes']}; "
            f"maximum height {f['max_height']}; total column heights {f['aggregate_height']}; "
            f"surface roughness {f['bumpiness']}; column heights {f['heights']}.")


class OllamaDecision:
    def __init__(self, model="nimble", endpoint="http://127.0.0.1:11434", timeout=120,
                 policy=POLICY, describe_option=describe):
        self.model, self.endpoint, self.timeout = model, endpoint.rstrip("/"), timeout
        self.policy, self.describe_option = policy, describe_option

    def request(self, state, options):
        if not 2 <= len(options) <= 26:
            raise ValueError("Decision API requires 2 to 26 choices")
        payload = {"model": self.model, "state": state, "keep_alive": "30m",
                   "questions": {"placement": {"type": "choice", "instructions": self.policy,
                                               "criteria": {m.id: self.describe_option(m) for m in options}}}}
        encoded = json.dumps(payload, allow_nan=False).encode()
        if len(encoded) > 65536:
            raise ValueError("Decision request exceeds 64 KiB")
        start = time.perf_counter()
        req = Request(self.endpoint + "/v1/systemone", data=encoded,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                raw = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read().decode()[:800]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama: {exc.reason}. Start Ollama and pull {self.model}.") from exc
        answer = raw.get("answers", {}).get("placement", {})
        ids = {m.id for m in options}
        probs = answer.get("probabilities", {})
        if (raw.get("model") != self.model or answer.get("type") != "choice"
                or answer.get("choice") not in ids or set(probs) != ids
                or any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                       for p in probs.values()) or abs(sum(probs.values()) - 1) > .02):
            raise RuntimeError("Ollama returned an invalid placement or probability distribution")
        concentration = answer.get("confidence")
        if not isinstance(concentration, (float, int)) or not math.isfinite(concentration) or not 0 <= concentration <= 1:
            raise RuntimeError("Ollama returned invalid confidence metadata")
        return {"request": payload, "response": raw,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "answer": answer, "options": [m.public() for m in options]}

    def choose(self, game):
        candidates = game.candidates()
        if not candidates:
            raise ValueError("No legal placement: game over")
        if len(candidates) == 1:
            move = candidates[0]
            return move, {"source": "forced", "choice": move.id, "latency_ms": 0,
                          "candidate_count": 1, "calls": [], "options": [move.public()]}
        # Every legal landing is considered. Shuffle to reduce systematic position bias.
        ordered = candidates[:]
        random.Random(game.seed * 10000 + game.pieces).shuffle(ordered)
        calls, observation = [], game.observation()
        if len(ordered) > 26:
            midpoint = len(ordered) // 2
            finalists = []
            for batch in (ordered[:midpoint], ordered[midpoint:]):
                call = self.request(observation, batch)
                calls.append(call)
                finalists.append(next(m for m in batch if m.id == call["answer"]["choice"]))
            ordered = finalists
        call = self.request(observation, ordered)
        calls.append(call)
        answer = call["answer"]
        move = next(m for m in candidates if m.id == answer["choice"])
        return move, {"source": "ollama", "model": self.model, "choice": move.id,
                      "latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
                      "candidate_count": len(candidates), "calls": calls,
                      "probabilities": answer["probabilities"], "concentration": answer["confidence"],
                      "options": call["options"]}
