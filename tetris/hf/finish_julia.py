"""After the serialized run, verify Julia's MPS batch-placement extension and benchmark it."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/"benchmarks/huggingface"
while not json.loads((OUT/"summary.json").read_text()).get("complete_all"):
    time.sleep(2)
args=subprocess.check_output(["ps","-p","15505","-o","args="],text=True).strip()
if "model_server.py --kind julia" not in args:
    raise RuntimeError("The original Julia process identity changed; refusing to stop another process")
os.kill(15505,signal.SIGTERM)
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    try:
        with urlopen("http://127.0.0.1:11438/v1/models",timeout=1):pass
    except OSError:break
    time.sleep(.2)
env={**os.environ,"HF_HOME":str(ROOT/"cache"),"HF_HUB_OFFLINE":"1"}
log=open("/private/tmp/tetris-julia-repaired-server.log","w")
server=subprocess.Popen([str(ROOT/"julia_runtime/.venv/bin/python"),str(ROOT/"model_server.py"),
    "--kind","julia","--device","mps","--model","julia-1:mps","--port","11438"],
    env=env,stdout=log,stderr=subprocess.STDOUT)
for _ in range(120):
    if server.poll() is not None:raise RuntimeError("Julia runtime startup failed")
    try:
        with urlopen("http://127.0.0.1:11438/v1/models",timeout=2) as r:metadata=json.load(r)
        break
    except OSError:time.sleep(1)
else:raise RuntimeError("Julia runtime startup timed out")
metadata["repo"]="SupersonicLabs/Julia-1";metadata["checkpoint"]="a85b127321d580d65176c89ced8273f305745d85"
meta=ROOT/"research/julia-1-mps-runtime.json";meta.write_text(json.dumps(metadata,indent=2))
print("Julia MPS batch placement repaired; checking CPU parity",flush=True)
parity_log=open("/private/tmp/tetris-julia-parity.log","w")
check=subprocess.run([str(ROOT/"julia_runtime/.venv/bin/python"),str(ROOT/"julia_parity.py")],
                     env=env,stdout=parity_log,stderr=subprocess.STDOUT)
if check.returncode:raise RuntimeError("Julia parity check failed; see its saved log")
bench_log=open("/private/tmp/tetris-julia-repaired-benchmark.log","w")
check=subprocess.run([str(ROOT.parent/".venv/bin/python"),str(ROOT/"benchmark_hf.py"),
    "--model","julia-1:mps","--label","Julia-1 · MPS FP32","--endpoint","http://127.0.0.1:11440",
    "--metadata",str(meta)],stdout=bench_log,stderr=subprocess.STDOUT)
if check.returncode:raise RuntimeError("Julia benchmark failed; see its saved log")
summary=json.loads((OUT/"summary.json").read_text())
summary["complete_all"]=True;summary["unavailable"]=[]
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
(OUT/"blocked.json").write_text(json.dumps([{"model":"julia","resolved":True,
    "initial_error":"Native packed batch stayed on CPU; explicit MPS batch placement adapter added."}],indent=2))
print("JULIA COMPLETE",flush=True)
while True:time.sleep(30)
