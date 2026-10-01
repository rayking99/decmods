"""Recompute reports from completed raw inference, audit, then export the site."""
from benchmark import ROOT, atomic, load_results, summarize
from run_models import SPECS


def main():
    results = load_results()
    assert results and results["status"] in ("complete", "complete_with_unavailable")
    assert all(m["status"] == "complete" for m in results["models"])
    expected = {m[0] for m in SPECS}
    observed = {m["name"] for m in results["models"]}
    unavailable = {m["name"] for m in results["unavailable"]}
    assert expected <= observed | unavailable, "A requested installed variant has not finished"
    results["method"] = (
        "Eight frozen states per game, two deterministic option orders. Every legal action reaches the native decision model. "
        "Shared 20-choice group tournament when needed, accepting all policy ties. No heuristic fallback. Serialized inference. "
        "Median/P95 use first-order trials, excluding warmups and states identical to warmup; repeated-order timings reported separately. "
        "Native caches may still reuse options or prefixes. One capped game per model/game with shared seed-42 starts; trajectories may diverge. "
        "Derived one-step/search metrics are supplied, so agreement measures declared-policy following rather than unaided planning or globally optimal play. "
        "BlockStar's original 20×18 board is one frozen case in two orders; seven cases and the capped replay are smaller fixtures. "
        "Model precision and native layouts differ. Model loading and downloads are excluded; complete decision latency includes every tournament call."
    )
    for model in results["models"]:
        model["summary"] = summarize(model)
    atomic(ROOT / "results.json",results)
    atomic(ROOT / "summary.json",{**{k:v for k,v in results.items() if k != "models"},
        "models":[{"name":m["name"],"label":m["label"],"status":m["status"],"metadata":m["metadata"],"games":m["summary"]} for m in results["models"]]})
    from verify import main as audit
    audit()
    from build_site_data import main as export
    export()


if __name__ == "__main__":
    main()
