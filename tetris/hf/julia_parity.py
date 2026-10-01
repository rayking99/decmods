"""Compare real Julia MPS outputs with the unchanged CPU reference on four inputs."""
import json
from pathlib import Path
import sys
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT/"models/julia"))
from julia import load_model
from benchmark import BENCH_POLICY,compact_state,compact_description,restore

engine=load_model(str(ROOT/"models/julia"),device="cpu",backend="torch",strict_encoding=True,
                  max_length=8192,head_length=2048)
suite=json.loads((ROOT.parent/"benchmarks/suite.json").read_text())
checks=[]
for index in (0,5,12,24):
    case=suite[index];game=restore(case["snapshot"]);moves=game.candidates()[:17]
    questions={"placement":{"type":"choice","instructions":BENCH_POLICY,
        "criteria":{m.id:compact_description(m) for m in moves}}}
    state=compact_state(game.observation())
    cpu=engine.predict(state=state,questions=questions)["answers"]["placement"]
    request=Request("http://127.0.0.1:11440/v1/systemone",data=json.dumps({"model":"julia-1:mps",
        "state":state,"questions":questions}).encode(),headers={"Content-Type":"application/json"})
    with urlopen(request,timeout=120) as r:gpu=json.load(r)["answers"]["placement"]
    difference=max(abs(cpu["probabilities"][k]-gpu["probabilities"][k]) for k in cpu["probabilities"])
    assert cpu["choice"]==gpu["choice"] and difference<1e-4
    checks.append({"case":case["id"],"choice":cpu["choice"],"max_probability_difference":difference,
                   "cpu":cpu,"mps":gpu})
output={"passed":True,"cases":len(checks),"max_probability_difference":max(x["max_probability_difference"] for x in checks),
        "checks":checks}
(ROOT/"research/julia-mps-parity.json").write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
