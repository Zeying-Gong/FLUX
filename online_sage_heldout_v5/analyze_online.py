from pathlib import Path
import sys,json,collections
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
r=Path(__file__).resolve().parent;o=Path(sys.argv[1]);p=o/"speed_090" if (o/"speed_090").exists() else o
v=json.loads((p/"verified.json").read_text());rows=json.loads((p/"actions.json").read_text());d=json.loads((p/"dynamics.json").read_text());contacts=json.loads((p/"contacts.json").read_text())
pairs=collections.defaultdict(lambda:dict(events=0,impulse_ns=0.))
for x in contacts:
 key=" | ".join(sorted([x["a"],x["b"]]));pairs[key]["events"]+=1;pairs[key]["impulse_ns"]+=x["impulse_ns"]
v["contact_pairs"]=dict(pairs);v["baseline_v4b"]=dict(job=60482,robot_path_m=3.121406523907192,final_gap_m=5.671328430632632,mean_gap_m=3.6941886264998556)
v["initial_pose_after_settling"]=rows[0]["pose_before"];v["mean_inference_ms"]=float(np.mean([x["model"]["latency_ms"] for x in rows]));v["comparison_caveats"]=["Different official dynamic Spot mesh and physical controller.","Body settles before start; actual initial pose recorded.","Human animation updated at 100Hz instead of 60Hz to match 500Hz physics; measured speed reported.","NavMesh audit only, with original physical mesh contacts resolving movement."]
(o/"analysis.json").write_text(json.dumps(v,indent=2))
cap=cv2.VideoCapture(str(p/"online.mp4"));sheet=Image.new("RGB",(1280,1720))
for i,sec in enumerate([.5,2,4,6]):
 cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,f=cap.read()
 if ok:sheet.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)),(0,i*430))
cap.release();sheet.save(o/"video_contact_sheet.jpg")
a=np.load(r/"nav_grid.npz");xs,ys=a["xs"],a["ys"];cell=float(a["cell"]);scale=5
im=Image.new("RGB",(len(xs)*scale,len(ys)*scale),(30,35,40));dr=ImageDraw.Draw(im)
def uv(p):return ((p[0]-xs[0])/cell*scale,(len(ys)-1-(p[1]-ys[0])/cell)*scale)
for tri in np.load(r/"native_nav_cache.npz")["triangles"]:
 dr.polygon([uv(p) for p in tri],fill=(45,105,65))
dr.line([uv(x["position"]) for x in d],fill=(255,150,0),width=6);dr.line([uv(x["target"]) for x in rows],fill=(155,30,180),width=4)
dr.rectangle((0,0,im.width,80),fill=(20,24,31));font=ImageFont.load_default(size=18)
dr.text((10,10),"Dynamic Spot: orange physical trajectory | purple human",fill="white",font=font)
dr.text((10,40),"NavMesh is audited; PhysX contacts determine motion (no pose clamp)",fill="white",font=font)
im.save(o/"online_navigation_map.png")
print(json.dumps({k:v[k] for k in ["status","robot_path_m","final_gap_m","max_body_tilt_deg","fell","nav_center_outside_samples"]},indent=2))
