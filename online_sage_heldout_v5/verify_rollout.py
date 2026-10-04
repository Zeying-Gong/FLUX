import pathlib,json,sys,math,hashlib
import cv2,numpy as np
p=pathlib.Path(sys.argv[1]);rows=json.loads((p/"actions.json").read_text());s=json.loads((p/"summary.json").read_text());d=json.loads((p/"dynamics.json").read_text());contacts=json.loads((p/"contacts.json").read_text());gait=json.loads((p/"gait.json").read_text())
assert rows and len(d)==len(rows)*25 and len(gait)==len(d)
previous=np.asarray(rows[0]["pose_before"]);length=0.;observations=set()
for i,x in enumerate(rows):
 a=np.array(x["pose_before"]);b=np.array(x["pose_after"]);np.testing.assert_allclose(a,previous,atol=1e-7)
 assert np.isfinite(b).all() and abs(x["sim_t"]-(i+1)*.25)<1e-7
 length+=np.linalg.norm(b[:2]-a[:2]);previous=b;observations.add(x["observation_sha256"])
 if not s["diagnostic"]:assert "<act_l" in x["model"]["raw"]
positions=np.array([x["position"] for x in d]);joints=np.array([x["joint_positions"] for x in d]);camera=np.array([x["camera_position"] for x in d]);tilts=np.rad2deg(np.arccos(np.clip([x["up_dot"] for x in d],-1,1)))
assert np.isfinite(positions).all() and np.isfinite(joints).all()
assert abs(d[-1]["physics_ticks"]-500*s["sim_seconds"])<=2,("physics clock mismatch",d[-1]["physics_ticks"],s["sim_seconds"])
assert joints.shape[1]==12 and np.max(np.ptp(joints,axis=0))>.05,"Joints did not move"
assert contacts and sum(x["impulse_ns"] for x in contacts)>0,"No physical contact evidence"
assert s["target_reached_route_end"] and s["min_target_clearance_m"]>=.3355
assert len(observations)>1
cap=cv2.VideoCapture(str(p/"online.mp4"));fps=cap.get(cv2.CAP_PROP_FPS);frames=0
while True:
 ok,f=cap.read()
 if not ok:break
 assert f.shape==(430,1280,3);frames+=1
cap.release();assert fps==30 and frames==len(rows)*30//4
gap=[np.linalg.norm(np.array(x["pose_after"][:2])-np.array(x["target"][:2])) for x in rows]
result=dict(status="VERIFIED_DYNAMIC_DIAGNOSTIC" if s["diagnostic"] else "VERIFIED_DYNAMIC_CLOSED_LOOP",control_steps=len(rows),physics_steps=d[-1]["physics_ticks"],video_frames=frames,video_seconds=frames/fps,distinct_live_observations=len(observations),robot_path_m=float(length),final_gap_m=float(gap[-1]),mean_gap_m=float(np.mean(gap)),body_height_range_m=[float(positions[:,2].min()),float(positions[:,2].max())],camera_height_range_m=[float(camera[:,2].min()),float(camera[:,2].max())],max_body_tilt_deg=float(tilts.max()),joint_ranges_rad=np.ptp(joints,axis=0).tolist(),physical_contact_events=len(contacts),total_contact_impulse_ns=sum(x["impulse_ns"] for x in contacts),nav_center_outside_samples=sum(not x["nav_center"] for x in d),nav_footprint_outside_samples=sum(not x["nav_footprint"] for x in d),fell=bool(tilts.max()>60 or positions[:,2].min()<.2),target_actual_mean_speed_mps=s["target_actual_mean_speed_mps"],video_sha256=hashlib.sha256((p/"online.mp4").read_bytes()).hexdigest(),limitation="Single derived heldout episode; official flat-ground locomotion controller; original mesh contacts, no root position projection; not full heldout benchmark.")
(p/"verified.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
