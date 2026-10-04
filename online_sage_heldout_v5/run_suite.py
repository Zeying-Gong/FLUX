"""Run one native walking tier; retain failure logs and exact source snapshot."""
import pathlib,os,subprocess,time,json,sys,tarfile,hashlib,ctypes
R=pathlib.Path(__file__).resolve().parent
O=pathlib.Path(sys.argv[1]); O.mkdir(parents=True,exist_ok=False)
P="/data/nas_ray/home/zeying.gong/algorithm/envs/lightnav/bin/python"
SIM="/data/nas_ray/home/zeying.gong/venvs/flux-isaac5/bin/python"
L="/data/nas_ray/home/zeying.gong/algorithm/repos/LightNav-0"
env=os.environ.copy(); env.update(PYTHONPATH=L+"/src",TOKENIZERS_PARALLELISM="false",PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
files=[p for p in R.rglob("*") if p.is_file() and not any(x in p.relative_to(R).parts for x in ["artifacts","runtime_deps","runtime_exts","wheels","__pycache__"])]; hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with tarfile.open(O/"source.tar.gz","w:gz") as t:
 for p in files: t.add(p,arcname=str(p.relative_to(R)))
 for p in (R.parent/"sage_utils").glob("*.py"): t.add(p,arcname="sage_utils/"+p.name)
(O/"source_hashes.json").write_text(json.dumps(hashes,indent=2))
(O/"runtime.txt").write_text(subprocess.check_output(["nvidia-smi"],text=True))
ctypes.CDLL("libEGL_nvidia.so.0")
(O/"nvidia_egl_icd.json").write_text(json.dumps({"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.194"}}))
log=(O/"policy.log").open("w"); ready=O/"policy_ready.json"
policy=subprocess.Popen([P,str(R/"policy_server.py"),"--ready-file",str(ready)],cwd=L,env={**env,"CUDA_VISIBLE_DEVICES":"1"},stdout=log,stderr=subprocess.STDOUT)
status={"status":"RUNNING","started_unix":time.time(),"speeds":[.9],"completed":[]}
def save(): (O/"suite_status.json").write_text(json.dumps(status,indent=2))
save()
try:
 deadline=time.monotonic()+600
 while not ready.exists():
  if policy.poll() is not None: raise RuntimeError("policy process exited")
  if time.monotonic()>deadline: raise TimeoutError("policy readiness")
  time.sleep(1)
 for speed in status["speeds"]:
  out=O/("speed_%03d"%round(speed*100)); print("RUN_SPEED",speed,flush=True)
  with (O/(out.name+".log")).open("w") as logsim:
   subprocess.run([SIM,str(R/"run_online.py"),"--allow-root","--gpu","0","--speed",str(speed),"--steps","60","--out",str(out)],cwd=str(R.parent),env={**env,"CUDA_VISIBLE_DEVICES":"0","OMNI_KIT_ACCEPT_EULA":"YES","VK_ICD_FILENAMES":str(O/"nvidia_egl_icd.json"),"LD_PRELOAD":"/opt/miniforge/envs/isaacray/lib/libstdc++.so.6"},stdout=logsim,stderr=subprocess.STDOUT,check=True,timeout=1200)
  subprocess.run([P,str(R/"verify_rollout.py"),str(out)],env=env,check=True)
  status["completed"].append(out.name); save()
 status["status"]="VERIFIED_DYNAMIC_SPOT_CLOSED_LOOP_SUITE"
except BaseException as e:
 status.update(status="FAILED",error=repr(e)); raise
finally:
 status["finished_unix"]=time.time(); save()
 policy.terminate()
 try: policy.wait(timeout=30)
 except subprocess.TimeoutExpired: policy.kill(); policy.wait()
 log.close()
