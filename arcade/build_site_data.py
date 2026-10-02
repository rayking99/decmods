"""Export compact, public replay data; omit duplicate raw API payloads."""
import json
import gzip
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "docs/data"


def save(name, data):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def clean_snapshot(snapshot):
    # Recorded frames do not need a 625-integer Python RNG state. Full raw traces retain it.
    return {k:v for k,v in snapshot.items() if k not in ("rng_state", "rng")}


def public_provenance(metadata):
    """Retain source identity and runtime facts without local model paths."""
    keys = ("repo", "checkpoint", "native_source", "source_commit", "backend", "version",
            "device", "dtype", "precision", "quantization", "mlx", "mlx_lm", "torch",
            "transformers", "ollaya", "cache", "prompt_layout", "versions", "option_order",
            "confidence", "max_length", "truncation", "loader_sha256", "joint_head_sha256")
    provenance = {key:metadata[key] for key in keys if key in metadata}
    runtimes = [{key:runtime[key] for key in keys if key in runtime}
                for runtime in metadata.get("models", [])]
    if runtimes:
        provenance["runtimes"] = runtimes
        if "backend" not in provenance:
            backend = next((runtime["backend"] for runtime in runtimes if "backend" in runtime),None)
            if backend is not None:
                provenance["backend"] = backend
    return provenance


def timing_quantile(values, fraction):
    """Use the same linearly interpolated percentile as the Tetris runner."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lo,hi = math.floor(position),math.ceil(position)
    value = ordered[lo] if lo == hi else ordered[lo]*(hi-position)+ordered[hi]*(position-lo)
    return round(value,1)


def trace_state_signature(trace):
    """The complete recorded API state, before option ordering or inference."""
    for call in trace.get("calls", []):
        request = call.get("request", {})
        if "state" in request:
            return json.dumps(request["state"],sort_keys=True,separators=(",", ":"))
    return None


def first_order_timing(model):
    """Compare matched first/repeated cohorts, excluding the warmed API state."""
    warm_state = trace_state_signature(model.get("warmup", {}))
    first,repeated,excluded = [],[],[]
    eligible_cases = set()
    for trial in model.get("trials", []):
        trace = trial.get("decision", {})
        latency = trace.get("latency_ms")
        if trial.get("error") or trial.get("order") != 0 or not isinstance(latency,(int,float)) or not math.isfinite(latency):
            continue
        if warm_state is not None and trace_state_signature(trace) == warm_state:
            excluded.append(trial["case"])
            continue
        first.append(latency)
        eligible_cases.add(trial["case"])
    for trial in model.get("trials", []):
        latency = trial.get("decision", {}).get("latency_ms")
        if (not trial.get("error") and trial.get("order") == 1 and trial.get("case") in eligible_cases
                and isinstance(latency,(int,float)) and math.isfinite(latency)):
            repeated.append(latency)
    return {"median_ms":round(statistics.median(first),1) if first else None,
            "p95_ms":timing_quantile(first,.95),
            "repeated_median_ms":round(statistics.median(repeated),1) if repeated else None,
            "repeated_p95_ms":timing_quantile(repeated,.95),
            "first_order_samples":len(first), "repeated_order_samples":len(repeated),
            "excluded_warmup_cases":excluded, "warmup_state_recorded":warm_state is not None,
            "method":"Order-0 full-decision timing; cases whose complete recorded API state equals warmup are excluded. Order-1 timing uses the same eligible cases. Loading and warmup are excluded; agreement still includes every frozen trial."}


def reference_game(game, registered_moves, **metadata):
    """Reconstruct genuine source moves without inventing per-slide inference."""
    from blockstar import translate
    initial, turns = game.snapshot(), []
    for piece,direction,distance in registered_moves:
        before = tuple(tuple(row) for row in game.board)
        after = translate(before,piece,direction,distance)
        metrics = game._metrics(after,distance)
        action = {"id":f"{piece}_{direction}_{distance}","text":f"Slide {piece} {direction} {distance} cells", "metrics":metrics,"after":after}
        game.board = [list(row) for row in after]
        game.turn += 1
        game.distance_moved += distance
        game.last_move = action["id"]
        turns.append({"action":action,"after":game.snapshot(),"decision":{"source":metadata["source"],"choice":action["id"],"latency_ms":None,"candidate_count":None,"calls":[]}})
    assert game.done, "A published source solution must reach the exact target"
    return {"game":"blockstar","seed":game.seed,"initial":initial,"moves":turns,"status":"solved",**metadata}


def main():
    tetris = ROOT.parent / "tetris/benchmarks"
    models = []
    summaries = []
    audit = {"decisions":0,"valid_decisions":0,"errors":0,"moves":0,"games":0,"models":0}
    labels = {"nimble":"Nimble 9B · Ollama Q8", "tev1:4b":"Tev1 4B · Ollama Q8", "tev1:0.8b":"Tev1 0.8B · Ollama Q8"}
    for folder in (tetris, tetris / "huggingface"):
        source = json.loads((folder / "results.json").read_text())
        for model in source["models"]:
            trials = model.get("trials", [])
            audit["decisions"] += len(trials)
            audit["valid_decisions"] += sum(not trial.get("error") and "decision" in trial for trial in trials)
            audit["errors"] += sum(bool(trial.get("error")) for trial in trials)
            audit["games"] += len(model.get("games", []))
            audit["moves"] += sum(len(game.get("moves", [])) for game in model.get("games", []))
            audit["models"] += 1
            name = "kev:4b-mps" if model["name"] == "kev-latest" else model["name"]
            label = labels.get(name,model.get("label",name))
            provenance = public_provenance(model.get("metadata", {}))
            protocol = model.get("protocol",model.get("metadata", {}).get("protocol"))
            summary = dict(model["summary"])
            entry = {"name":name, "label":label, "status":model["status"], "summary":summary,
                     "provenance":provenance}
            if provenance.get("backend"):
                entry["runtime"] = provenance["backend"]
            if protocol is not None:
                entry["protocol"] = protocol
            if name == "clef-flash:mlx-4bit":
                fresh = first_order_timing(model)
                summary["all_order_timing"] = {key:summary.get(key) for key in ("mean_placement_ms","median_placement_ms","p95_placement_ms")}
                summary["median_placement_ms"] = fresh["median_ms"]
                summary["p95_placement_ms"] = fresh["p95_ms"]
                summary["repeated_median_ms"] = fresh["repeated_median_ms"]
                summary["timing_note"] = fresh["method"]
                summary["fresh_timing"] = fresh
                entry["fresh_timing"] = fresh
            summaries.append(entry)
            games = []
            for game in model["games"]:
                moves = []
                for move in game["moves"]:
                    decision = move["decision"]
                    moves.append({"move":move["move"], "after":move["after"],
                        "decision":{"latency_ms":decision["latency_ms"], "candidate_count":decision["candidate_count"],
                                    "calls":[{"latency_ms":c["latency_ms"], "choice":c["answer"]["choice"]} for c in decision["calls"]], "choice":decision["choice"],
                                    "probabilities":decision.get("probabilities", {}), "source":decision["source"]}})
                games.append({"seed":game["seed"], "status":game["status"], "moves":moves})
            replay = {"name":name, "label":label, "games":games,"provenance":provenance}
            if protocol is not None:
                replay["protocol"] = protocol
            models.append(replay)
    cache = json.loads((tetris / "huggingface/cache-analysis.json").read_text())
    for model in summaries:
        if model["name"] == "clm:8b-mps":
            model["summary"]["median_placement_ms"] = cache["first_order_median_ms"]
            model["summary"]["p95_placement_ms"] = cache["first_order_p95_ms"]
            model["summary"]["timing_note"] = "First-order timing; repeated embedding cache hits excluded from principal median."
            model["fresh_timing"] = {"median_ms":cache["first_order_median_ms"], "p95_ms":cache["first_order_p95_ms"],
                                     "repeated_median_ms":cache["repeated_order_median_ms"]}
            model["summary"]["fresh_timing"] = model["fresh_timing"]
    save("tetris-summary.json", {"status":"complete" if all(model["status"] == "complete" for model in summaries) else "running", "models":summaries,
        "method":"32 frozen boards in two orders; separate capped games with seeds 42 and 123. Agreement follows a disclosed one-step policy, not optimal Tetris play. Earlier matched run, separately timed.",
        "hardware":"Apple M3 Max, 128 GB unified memory", "cache_analysis":cache, "audit":audit})
    save("tetris-replays.json", {"models":models})
    from benchmark import load_results, summarize
    results = load_results()
    if results is not None:
        summary = {k:v for k,v in results.items() if k != "models"}
        summary["models"] = []
        for model in results["models"]:
            metadata = model.get("metadata", {})
            summary["models"].append({"name":model["name"], "label":model["label"], "status":model["status"],
                "games":summarize(model),
                "provenance":public_provenance(metadata)})
        save("arcade-summary.json", summary)
        replays = []
        for model in results["models"]:
            games = []
            for game in model["games"]:
                moves = []
                for move in game["moves"]:
                    decision = move["decision"]
                    moves.append({"action":move["action"], "after":clean_snapshot(move["after"]),
                        "decision":{"choice":decision["choice"], "latency_ms":decision["latency_ms"],
                            "candidate_count":decision["candidate_count"],
                            "calls":[{"latency_ms":c["latency_ms"], "choice":c["answer"]["choice"]} for c in decision["calls"]],
                            "probabilities":decision.get("probabilities", {}), "source":decision["source"]}})
                games.append({**{k:v for k,v in game.items() if k not in ("initial", "moves")},
                              "initial":clean_snapshot(game["initial"]), "moves":moves})
            replays.append({"name":model["name"], "label":model["label"], "games":games})
        source = json.loads((ROOT / "blockstar-source.json").read_text())
        from blockstar import BlockStar
        reference = []
        for trial in source["upstream_measured_baseline"]["trials"]:
            reference.append(reference_game(BlockStar(trial["seed"]),trial["moves"],
                label=f"Legacy original solver · seed {trial['seed']}",scope="full-original-source-baseline",
                source="original-stochastic-solver",elapsed_s=trial["elapsed_s"],source_commit=source["commit"]))
        replays.append({"name":"blockstar:original-solver","label":"BlockStar legacy solver · reference","games":reference})
        if (ROOT / "blockstar-update-source.json").exists():
            updated_source = json.loads((ROOT / "blockstar-update-source.json").read_text())
            from blockstar_update import make_original,make_large_yard
            reference = []
            for trial in updated_source.get("replay_reference",[]):
                game = (make_original if trial["case_id"] == "updated-default" else make_large_yard)(trial.get("seed",42))
                reference.append(reference_game(game,trial["moves"],
                    label=f"Updated {trial['case_id']} · {trial['slide_count']} slides",scope="updated-official-source-baseline",
                    source="recorded-native-discovery",elapsed_s=trial.get("discovery_elapsed_seconds"),
                    discovery_elapsed_seconds=trial.get("discovery_elapsed_seconds"),
                    replay_verification_s=trial.get("certificate_replay_elapsed_seconds"),
                    source_commit=updated_source["commit"],kind=trial["kind"]))
            if reference:
                replays.append({"name":"blockstar:updated-solver","label":"BlockStar updated solver · reference","games":reference})
            save("blockstar-update-source.json",updated_source)
        from benchmark_update import load_results as load_updated_results, summarize as summarize_updated
        updated = load_updated_results()
        if updated:
            summary = {k:v for k,v in updated.items() if k != "models"}
            summary["models"] = [{"name":m["name"],"label":m["label"],"status":m["status"],"summary":summarize_updated(m)} for m in updated["models"]]
            save("blockstar-update-summary.json",summary)
            suite = json.loads((ROOT / "updated/suite.json").read_text())
            case_map = {c["id"]:c for c in suite}
            for model in updated["models"]:
                replay = next((m for m in replays if m["name"] == model["name"]),None)
                if not replay:
                    replay = {"name":model["name"],"label":model["label"],"games":[]}
                    replays.append(replay)
                for trial in model["trials"]:
                    if "decision" not in trial:
                        continue
                    trace = trial["decision"]
                    replay["games"].append({"game":"blockstar","seed":42,"label":f"{trial['case']} · order {trial['order']+1}",
                        "scope":"updated-official-one-step-probe","source_commit":updated["source_commit"],
                        "initial":case_map[trial["case"]]["snapshot"],"status":"one_step_probe",
                        "moves":[{"action":trial["action"],"after":trial["after"],"decision":{"choice":trace["choice"],
                            "source":trace["source"],"latency_ms":trace["latency_ms"],"candidate_count":trace["candidate_count"],
                            "calls":[{"choice":c["answer"]["choice"],"latency_ms":c["latency_ms"]} for c in trace["calls"]],
                            "probabilities":trace.get("probabilities",{})}}]})
            if updated["status"] in ("complete","complete_with_unavailable"):
                with open(DATA / "blockstar-update-results.json.gz","wb") as output:
                    with gzip.GzipFile(filename="",mode="wb",fileobj=output,mtime=0) as stream:
                        stream.write(json.dumps(updated,ensure_ascii=False,allow_nan=False,separators=(",", ":")).encode())
            save("blockstar-update-suite.json",suite)
            if (ROOT / "updated/verification.json").exists():
                save("blockstar-update-verification.json",json.loads((ROOT / "updated/verification.json").read_text()))
        save("arcade-replays.json", {"models":replays})
        if results["status"] in ("complete", "complete_with_unavailable"):
            # Deterministic compression preserves all public raw evidence,
            # and keeps the large, repetitive boards below GitHub's file limit.
            with open(DATA / "arcade-results.json.gz", "wb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
                    stream.write(json.dumps(results, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
    else:
        save("arcade-summary.json", {"status":"pending", "models":[], "hardware":"Apple M3 Max, 128 GB unified memory"})
        save("arcade-replays.json", {"models":[]})
    for file in ("blockstar-source.json", "verification.json", "suite.json"):
        if (ROOT / file).exists():
            save(file, json.loads((ROOT / file).read_text()))
    print("Exported site evidence to", DATA)


if __name__ == "__main__":
    main()
