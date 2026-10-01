"""Serialized native-runtime launch and benchmark; checkpoint downloads are done beforehand."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "benchmarks/huggingface"
env = {**os.environ, "HF_HOME": str(ROOT/"cache"), "HF_HUB_OFFLINE":"1", "HF_HUB_DISABLE_PROGRESS_BARS":"1"}
server_python = str(ROOT/".venv/bin/python")
bench_python = str(ROOT.parent/".venv/bin/python")
servers = []


def get(url, body=None):
    req = Request(url,data=json.dumps(body).encode() if body is not None else None,
                  headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=300) as response:return json.load(response)


def wait_ready(port, process=None):
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if process and process.poll() is not None:raise RuntimeError(f"Runtime exited: {process.returncode}")
        try:return get(f"http://127.0.0.1:{port}/v1/models")
        except (URLError,OSError):time.sleep(1)
    raise RuntimeError("Runtime startup timeout")


def publish_phase(message):
    path=OUT/"summary.json"
    data=json.loads(path.read_text());data["status"]="running";data["active"]=message
    data["updated"]=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data,indent=2))
    print(message,flush=True)


def native(kind,model,port,python=server_python):
    log=open(f"/private/tmp/tetris-{kind}-server.log","a")
    command=[python,str(ROOT/"model_server.py"),"--kind",kind,"--device","mps","--model",model,"--port",str(port)]
    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    servers.append(process)
    return wait_ready(port,process)


def run(model,label,metadata):
    meta=ROOT/"research"/(model.replace(":","-")+"-runtime.json")
    meta.write_text(json.dumps(metadata,indent=2))
    log=open(f"/private/tmp/tetris-{model.replace(':','-')}-benchmark.log","w")
    command=[bench_python,str(ROOT/"benchmark_hf.py"),"--model",model,"--label",label,
             "--endpoint","http://127.0.0.1:11440","--metadata",str(meta)]
    process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError(f"Benchmark exited {process.returncode}; see {log.name}")
    data=json.loads((OUT/"results.json").read_text())
    measured=next(m for m in data["models"] if m["name"]==model)
    print(label,json.dumps(measured["summary"]),flush=True)


def provenance(repo):
    return {"repo":repo,"checkpoint":json.loads((ROOT/"research"/(repo.replace('/','__')+'.json')).read_text())["sha"]}


def main():
    # Do not compete for the GPU with the already-running MPS benchmark.
    while True:
        data=json.loads((OUT/"results.json").read_text())
        kev=next((m for m in data["models"] if m["name"]=="kev-latest"),None)
        if kev and kev["status"]=="complete":break
        time.sleep(2)
    blocked=[]
    for kind in ("julia","winnow","kev-mlx","laya-multilingual","clm","lev"):
        publish_phase(f"Loading {kind} · inference remains serialized")
        try:
            if kind=="julia":
                meta=wait_ready(11438)
                meta.update(provenance("SupersonicLabs/Julia-1"))
                run("julia-1:mps","Julia-1 · MPS FP32",meta)
            elif kind=="winnow":
                # A warm request loads the GGUF; capture the reported engine device afterwards.
                sys.path.insert(0,str(ROOT.parent))
                from benchmark import BenchmarkDecision
                from game import Game
                warm=Game(98765);warm.queue[0]="O"
                _,trace=BenchmarkDecision("winnow:e4b","http://127.0.0.1:11440",timeout=300).choose(warm)
                meta=get("http://127.0.0.1:11435/api/ps")
                actual=next(m for m in meta["models"] if m["name"]=="winnow:e4b")
                if actual["device"]!="metal":raise RuntimeError(f"Winnow did not select Metal: {actual['device']}")
                meta.update(provenance("EldanRing/Winnow-E4B"))
                meta["ollaya"]="0.8.0"
                meta["manifest"]=json.loads((ROOT/"vendor/ollaya/registry/v2/library/winnow/manifests/e4b").read_text())
                meta["loader_probe"]=trace
                run("winnow:e4b","Winnow E4B · Metal Q8",meta)
            elif kind=="kev-mlx":
                log=open("/private/tmp/tetris-kev-mlx-server.log","w")
                kev_env={**env,"KEV_BACKEND":"mlx","KEV_DTYPE":"bf16","KEV_PREFIX_CACHE":"4"}
                process=subprocess.Popen([str(ROOT/"vendor/kev/.venv/bin/python"),"-m","kev.serve",
                    "--run",str(ROOT/"models/kev"),"--port","11439"],cwd=ROOT/"vendor/kev",env=kev_env,
                    stdout=log,stderr=subprocess.STDOUT)
                servers.append(process);meta=wait_ready(11439,process)
                meta.update(provenance("jaredpalmer/kev-4b"));meta["native_source"]="90512f1c517d977741f2104470a40635408236c9"
                run("kev:4b-mlx","Kev 4B · MLX BF16",meta)
            else:
                model,port,label,repo={
                    "laya-multilingual":("laya:multilingual-mps",11441,"Laya multilingual · MPS","convaiinnovations/laya"),
                    "clm":("clm:8b-mps",11443,"CLM 8B · MPS BF16","Contrastive-LM/CLM-v0.1-8B"),
                    "lev":("lev:4b-mps",11442,"Lev 4B · MPS BF16","interfaze-ai/lev")}[kind]
                if kind=="lev":
                    # Seed only this workspace's HF cache from the exact downloaded base snapshot.
                    base=json.loads((ROOT/"research/lev-base-source.json").read_text())
                    cache=ROOT/"cache/hub"/("models--"+base["repo"].replace("/","--"))
                    (cache/"snapshots").mkdir(parents=True,exist_ok=True);(cache/"refs").mkdir(exist_ok=True)
                    snapshot=cache/"snapshots"/base["sha"]
                    if not snapshot.exists():snapshot.symlink_to(ROOT/"models/lev-base",target_is_directory=True)
                    (cache/"refs/main").write_text(base["sha"])
                meta=native(kind,model,port);meta.update(provenance(repo))
                run(model,label,meta)
        except Exception as exc:
            entry={"model":kind,"reason":str(exc),"at":datetime.now(timezone.utc).isoformat()}
            blocked.append(entry);print("UNAVAILABLE",json.dumps(entry),flush=True)
            (OUT/"blocked.json").write_text(json.dumps(blocked,indent=2))
    data=json.loads((OUT/"summary.json").read_text())
    data["status"]="complete";data["active"]=None;data["complete_all"]=True;data["unavailable"]=blocked
    (OUT/"summary.json").write_text(json.dumps(data,indent=2))
    print("SERIAL BENCHMARKS FINISHED",flush=True)
    # Keep the loaded native APIs available to the user's local gateway after this tool call.
    while True:time.sleep(30)


if __name__=="__main__":main()
