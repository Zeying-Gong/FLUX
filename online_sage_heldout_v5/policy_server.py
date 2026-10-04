"""Local-only LightNav adapter; receives RGB, never target/world state."""
import argparse,json,base64,io,pathlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
from PIL import Image
from lightnav.tracking import build_tracking_agent
p=argparse.ArgumentParser(); p.add_argument("--port",type=int,default=18763); p.add_argument("--ready-file",required=True); args=p.parse_args()
agent=build_tracking_agent(model_path="/data/nas_ray/home/zeying.gong/algorithm/repos/LightNav-0/checkpoints/LightNav-0",backend="hf",device="cuda",task_key="trackvla",max_new_tokens=16)
instruction=None
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  global instruction
  try:
   data=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
   if self.path=="/reset":
    if set(data)!={"instruction"}: raise ValueError("Unexpected reset fields")
    instruction=data["instruction"]; agent.reset(instruction=instruction); result={"ready":True}
   elif self.path=="/step":
    if instruction is None or set(data)!={"rgb"}: raise ValueError("RGB-only episode reset required")
    rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(data["rgb"]))).convert("RGB")); agent.observe(rgb)
    wp,raw,ms=agent.predict_waypoints(instruction,task_type="tracking"); wp=np.asarray(wp)
    if wp.shape!=(10,3) or not np.isfinite(wp).all(): raise ValueError("Invalid predicted action")
    result=dict(waypoints=wp.tolist(),raw=raw,latency_ms=float(ms))
   else: raise ValueError("Unknown endpoint")
   payload=json.dumps(result).encode(); self.send_response(200)
  except Exception as e:
   payload=json.dumps({"error":str(e)}).encode(); self.send_response(500)
  self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(payload))); self.end_headers(); self.wfile.write(payload)
 def log_message(self,*args): pass
server=HTTPServer(("127.0.0.1",args.port),Handler)
pathlib.Path(args.ready_file).write_text(json.dumps(dict(port=args.port,backend="hf",task="tracking")))
print("POLICY_READY",flush=True); server.serve_forever()
