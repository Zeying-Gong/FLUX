"""SAGE online dynamic Spot rollout with official joint-drive locomotion policy.
Policy receives live RGB only. Target uses native root motion along a clearance-checked derived heldout route.
"""
import argparse,pathlib,json,sys,math,time,base64,urllib.request,hashlib
import numpy as np
from control import path_at_distance
from embodied_control import waypoint_delta,action_target,footprint,navigate
p=argparse.ArgumentParser(); p.add_argument("--gpu",type=int,default=6); p.add_argument("--speed",type=float,default=.9); p.add_argument("--steps",type=int,default=60); p.add_argument("--diagnostic",action="store_true"); p.add_argument("--port",type=int,default=18763); p.add_argument("--out",required=True)
args,kitargs=p.parse_known_args(); assert abs(args.speed-.9)<1e-8,"Only nominal native walking tier 0.9 is supported"; sys.argv=[sys.argv[0]]+kitargs
ROOT=pathlib.Path(__file__).resolve().parent; OUT=pathlib.Path(args.out); OUT.mkdir(parents=True,exist_ok=True); case=json.loads((ROOT/"case_v4.json").read_text())
profile=json.loads((ROOT/"embodiment.json").read_text()); camera=profile["camera"]
ASSETS="/data/nas_ray/md-ai-kit/md_ai_kit.bot/isaacsim_assets/5.0"
sys.argv.append("--/persistent/isaac/asset_root/default="+ASSETS)
sys.argv += ["--ext-folder", "/opt/miniforge/envs/isaacray/lib/python3.11/site-packages/isaacsim/extscache"]
sys.argv += ["--ext-folder", str(ROOT/"runtime_exts")]
sys.path.insert(0,str(ROOT/"runtime_deps"))
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"renderer":"RayTracedLighting","active_gpu":args.gpu,"physics_gpu":0,"multi_gpu":False},experience=str(ROOT/"online.kit"))
import carb,omni.usd,omni.replicator.core as rep,omni.timeline,omni.physx
from pxr import Usd,UsdGeom,UsdLux,UsdPhysics,PhysxSchema,PhysicsSchemaTools,Gf
from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy
from isaacsim.core.utils.rotations import quat_to_rot_matrix,quat_to_euler_angles
from isaacsim.core.api import World
from isaacsim.replicator.agent.core.settings import AssetPaths,PrimPaths
from isaacsim.replicator.agent.core.stage_util import CharacterUtil
from PIL import Image,ImageDraw
import io
settings=carb.settings.get_settings(); settings.set(AssetPaths.DEFAULT_CHARACTER_PATH,ASSETS+"/Isaac/People/Characters"); settings.set(AssetPaths.DEFAULT_BIPED_ASSET_PATH,ASSETS+"/Isaac/People/Characters/Biped_Setup.usd")
omni.usd.get_context().open_stage(str(ROOT/(case["scene_id"]+".usda")))
for _ in range(10): app.update()
stage=omni.usd.get_context().get_stage(); print("STAGE_READY",flush=True)
colliders=0
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/scene_collision")):
 if prim.IsA(UsdGeom.Mesh):
  UsdPhysics.CollisionAPI.Apply(prim); colliders+=1
print("COLLISION_MESHES",colliders,flush=True); assert colliders>0
from navigation import NativeNavigation
native_nav=NativeNavigation(ROOT)
nav_validation=native_nav.validate_case(case)
(OUT/"navmesh").mkdir(exist_ok=True)
(OUT/"navmesh/validated_paths.json").write_text(json.dumps(nav_validation,indent=2))
def native_on_nav(p):return native_nav.contains(p)
light=UsdLux.DomeLight.Define(stage,"/World/OnlineLight"); light.CreateIntensityAttr(900.)
world=World(physics_dt=.002,rendering_dt=1/100,stage_units_in_meters=1.0)
sys.path.insert(0,str(ROOT.parent/"sage_utils"))
from people_utils import spawn_character,load_default_skeleton_and_animations,bind_animation_graph_to_characters
load_default_skeleton_and_animations(app)
start=np.asarray(case["target_points"][0]); cpath=spawn_character(app,0,case["character_asset"],start)
bind_animation_graph_to_characters(app)
pose=np.array([*case["robot_start"][:2],case["robot_yaw"]]); robot_z=.7
spot=SpotFlatTerrainPolicy(prim_path="/World/OnlineRobot",usd_path=profile["body"]["usd"],position=np.array([pose[0],pose[1],robot_z]),orientation=np.array([math.cos(pose[2]/2),0,0,math.sin(pose[2]/2)]))
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/OnlineRobot")):
 if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
command=np.zeros(3);physics_ticks=0;contact_events=[]
def contact_report(headers,data):
 for h in headers:
  a=str(PhysicsSchemaTools.intToSdfPath(h.actor0));b=str(PhysicsSchemaTools.intToSdfPath(h.actor1))
  if "OnlineRobot" not in a+b:continue
  impulse=sum(float(np.linalg.norm(list(data[i].impulse))) for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data))
  if impulse>1e-9:contact_events.append(dict(tick=physics_ticks,a=a,b=b,impulse_ns=impulse))
contact_subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact_report)
def physics_control(dt):
 global physics_ticks
 spot.forward(dt,command);physics_ticks+=1
cams=[]; rgbs=[]
for name in ["Policy","Chase"]:
 cam=UsdGeom.Camera.Define(stage,"/World/"+name); op=cam.AddTransformOp(); cw,ch,hfov=(camera["width"],camera["height"],camera["hfov_deg"]) if name=="Policy" else (640,360,90);cam.CreateFocalLengthAttr(18./math.tan(math.radians(hfov/2)));cam.CreateHorizontalApertureAttr(36.);cam.CreateVerticalApertureAttr(36.*ch/cw);cam.CreateClippingRangeAttr(Gf.Vec2f(.05,100));rp=rep.create.render_product(cam.GetPath(),(cw,ch)); rgb=rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp); cams.append(op); rgbs.append(rgb)
def place_robot():
 global pose
 pos,q=spot.robot.get_world_pose();rot=quat_to_rot_matrix(q);yaw=quat_to_euler_angles(q)[2];pose=np.array([pos[0],pos[1],yaw])
 forward=rot[:,0];up=rot[:,2];eye=pos+rot@np.array([.24,0,.24]);chase=np.array([pos[0],pos[1],2.4])-np.array([math.cos(yaw),math.sin(yaw),0])*2.4
 cams[0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*(eye+forward)),Gf.Vec3d(*up)).GetInverse())
 cams[1].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*chase),Gf.Vec3d(*(pos+np.array([math.cos(yaw),math.sin(yaw),0]))),Gf.Vec3d(0,0,1)).GetInverse())
place_robot()
camprim=UsdGeom.Camera(stage.GetPrimAtPath("/World/Policy"))
calibration=dict(width=camera["width"],height=camera["height"],camera_height_m=camera["height_m"],forward_offset_m=camera["forward_offset_m"],hfov_deg=math.degrees(2*math.atan(camprim.GetHorizontalApertureAttr().Get()/(2*camprim.GetFocalLengthAttr().Get()))),vfov_deg=math.degrees(2*math.atan(camprim.GetVerticalApertureAttr().Get()/(2*camprim.GetFocalLengthAttr().Get()))),body_asset=profile["body"]["usd"],mount_translation_body_m=[.24,0,.24],height_semantics="nominal; dynamically follows full body pose")
assert abs(calibration["hfov_deg"]-camera["hfov_deg"])<1e-5
(OUT/"camera_calibration.json").write_text(json.dumps(calibration,indent=2))
world.reset()
spot.initialize();spot.robot.set_joint_positions(spot.default_pos);spot.robot.set_joint_velocities(np.zeros(12))
world.add_physics_callback("spot_locomotion",physics_control)
settle=[]
for i in range(200):
 world.step(render=True);place_robot()
 pos,q=spot.robot.get_world_pose()
 settle.append(dict(frame=i,ticks=physics_ticks,pos=pos.tolist(),q=q.tolist(),joints=spot.robot.get_joint_positions().tolist(),action=spot.action.tolist(),dt=world.get_physics_dt()))
 if i%20==0:print("SETTLING",settle[-1],"defaults",list(spot.default_pos),flush=True)
(OUT/"settling_trace.json").write_text(json.dumps(settle))
Image.fromarray(np.asarray(rgbs[1].get_data())[:,:,:3].copy()).save(OUT/"settled_chase.png")
(OUT/"settling_contacts.json").write_text(json.dumps(contact_events))

pos,q=spot.robot.get_world_pose()
if not .3<pos[2]<.8:
 print("FAILED_SETTLING",pos.tolist(),flush=True);app.close();sys.exit(2)
assert quat_to_rot_matrix(q)[2,2]>.85,("settling tilt",q.tolist())
(OUT/"settled.json").write_text(json.dumps(dict(position=pos.tolist(),orientation=q.tolist(),joint_names=spot.robot.dof_names,default_pos=list(spot.default_pos),policy_dt=spot._dt,decimation=spot._decimation,physics_ticks=physics_ticks),indent=2))
import omni.anim.graph.core as ag
chars=[x for x in CharacterUtil.get_characters_in_stage() if str(x.GetPath()).startswith(cpath)]
print("CHARACTERS",[str(x.GetPath()) for x in chars],flush=True); assert len(chars)==1
character=ag.get_character(str(chars[0].GetPath())); assert character is not None
nav_data=np.load(ROOT/"nav_grid.npz");xs=nav_data["xs"];ys=nav_data["ys"];walk=nav_data["walk"];clearance=nav_data["clearance"];cell=float(nav_data["cell"])
def nav_clear(p):
 ix=int(round((p[0]-xs[0])/cell));iy=int(round((p[1]-ys[0])/cell))
 return float(clearance[iy,ix]) if 0<=iy<len(ys) and 0<=ix<len(xs) else 0.
def on_nav(p):return nav_clear(p)>=.3+cell*.71 and native_on_nav(p)
assert native_on_nav(pose)
point,yaw,_=path_at_distance(case["target_points"],0);a=yaw+math.pi/2
character.set_world_transform(carb.Float3(*map(float,point)),carb.Float4(0,0,math.sin(a/2),math.cos(a/2)))
character.set_variable("PathPoints",[carb.Float3(*map(float,p)) for p in case["target_points"]])
character.set_variable("Action","None");character.set_variable("Walk",0.)
def target_state():
 p=carb.Float3(0,0,0);q=carb.Float4(0,0,0,1);character.get_world_transform(p,q)
 return np.array(list(p))
for _ in range(10):world.step(render=True);place_robot()
target=target_state();initial_target=target.copy();initial_pose=pose.copy()
first=np.asarray(rgbs[0].get_data())[:,:,:3].copy();Image.fromarray(first).save(OUT/"first_frame.png")
def call(path,data):
 req=urllib.request.Request("http://127.0.0.1:%d/%s"%(args.port,path),data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
 return json.load(urllib.request.urlopen(req,timeout=90))
if not args.diagnostic:call("reset",dict(instruction=case["instruction"]))
from PIL import ImageFont,ImageOps
font=ImageFont.load_default(size=16)
frames=[];records=[];gait=[];dynamics=[];sim_t=0.;blocked=0;stop_at=None;target_path=0.;blend=0.
start_physics_ticks=physics_ticks;contact_events.clear()
character.set_variable("Action","Walk")
for step in range(args.steps):
 rgb=np.asarray(rgbs[0].get_data())[:,:,:3].copy()
 if args.diagnostic:response=dict(waypoints=[[.125,0,.05]]*10,latency_ms=0,raw="DIAGNOSTIC_COMMAND")
 else:
  b=io.BytesIO();Image.fromarray(rgb).save(b,format="PNG");response=call("step",dict(rgb=base64.b64encode(b.getvalue()).decode()))
 delta_cmd=waypoint_delta(response["waypoints"][0]);v,vy,w=delta_cmd/.25;command=np.array([v,vy,w]);step_heading=pose[2]
 before=pose.copy();hits=0;nav_hits=0;slides=0;collision_paths=set()
 for sub in range(25):
  if np.linalg.norm(target[:2]-np.asarray(case["target_points"][-1])[:2])<.2 and stop_at is None:stop_at=sim_t
  blend=float(np.clip(blend+(1 if stop_at is None else -1)/25,0,1));character.set_variable("Walk",blend)
  if stop_at is not None and blend==0:character.set_variable("Action","None")
  old_target=target.copy();world.step(render=True);place_robot();sim_t+=1/100;target=target_state();target_path+=float(np.linalg.norm(target[:2]-old_target[:2]));cl=nav_clear(target)
  assert cl>=.3+cell*.71 and native_on_nav(target),"Native character left clearance-safe route: "+str((target.tolist(),cl))
  pos,q=spot.robot.get_world_pose();rot=quat_to_rot_matrix(q);joints=spot.robot.get_joint_positions();vel=spot.robot.get_linear_velocity()
  dynamics.append(dict(t=sim_t,physics_ticks=physics_ticks-start_physics_ticks,position=pos.tolist(),orientation=q.tolist(),up_dot=float(rot[2,2]),joint_positions=joints.tolist(),joint_velocities=spot.robot.get_joint_velocities().tolist(),linear_velocity=vel.tolist(),angular_velocity=spot.robot.get_angular_velocity().tolist(),camera_position=(pos+rot@np.array([.24,0,.24])).tolist(),nav_center=native_on_nav(pose),nav_footprint=all(native_on_nav(p) for p in footprint(pose))))
  if not dynamics[-1]["nav_footprint"]:nav_hits+=1
  assert np.isfinite(pos).all() and np.isfinite(joints).all(),"Nonfinite dynamics"
  feet={}
  for joint in ["L_Foot","R_Foot","L_ToeBase","R_ToeBase"]:
   pp=carb.Float3(0,0,0);qq=carb.Float4(0,0,0,1)
   assert character.get_joint_transform(joint,pp,qq),joint
   feet[joint]=list(pp)
  gait.append(dict(t=sim_t,pos=target.tolist(),feet=feet,clearance_m=cl))
  if len(frames)<int(round(sim_t*100))*30//100:
   aa=np.asarray(rgbs[0].get_data())[:,:,:3].copy();bb=np.asarray(rgbs[1].get_data())[:,:,:3].copy();f=Image.new("RGB",(1280,430),(16,20,26));panel=ImageOps.contain(Image.fromarray(aa),(640,360));f.paste(panel,((640-panel.width)//2,70+(360-panel.height)//2));f.paste(Image.fromarray(bb),(640,70));draw=ImageDraw.Draw(f)
   draw.text((12,8),("DIAGNOSTIC (no model)" if args.diagnostic else "ONLINE LightNav")+" | Dynamic Spot / joint drives | native human gait ~0.9 m/s",font=font,fill="white")
   draw.text((12,36),"t %.2fs | cmd fwd %.2f / lateral %.2f m/s, yaw %.2f rad/s | gap %.2fm"%(sim_t,v,vy,w,np.linalg.norm(pose[:2]-target[:2])),font=font,fill="white");frames.append(np.asarray(f))
 blocked+=int(hits>0 or nav_hits>0)
 records.append(dict(step=step,observation_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),sim_t=sim_t,pose_before=before.tolist(),pose_after=pose.tolist(),target=target.tolist(),v=float(v),vy=float(vy),w=float(w),waypoint_delta=delta_cmd.tolist(),slide_substeps=slides,executed_distance_m=float(np.linalg.norm(pose[:2]-before[:2])),executed_v_mps=float(np.linalg.norm(pose[:2]-before[:2]))/.25,collision_paths=sorted(collision_paths),blocked_substeps=hits,nav_blocked_substeps=nav_hits,model=response))
 (OUT/"actions.json").write_text(json.dumps(records,indent=2));print("ACTION",step,pose.tolist(),"target",target.tolist(),"hits",hits,"nav_hits",nav_hits,flush=True)
 if stop_at is not None and sim_t-stop_at>=.5:break
import subprocess
enc=subprocess.Popen(["/data/nas_ray/home/zeying.gong/algorithm/envs/lightnav/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1280x430","-r","30","-i","-","-an","-c:v","libx264","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(OUT/"online.mp4")],stdin=subprocess.PIPE)
for f in frames:enc.stdin.write(f.tobytes())
enc.stdin.close();assert enc.wait()==0
(OUT/"gait.json").write_text(json.dumps(gait))
(OUT/"dynamics.json").write_text(json.dumps(dynamics))
(OUT/"contacts.json").write_text(json.dumps(contact_events))
summary=dict(status="DEBUG_PASS" if args.diagnostic else "ROLLOUT_COMPLETE",closed_loop=not args.diagnostic,diagnostic=args.diagnostic,steps=len(records),sim_seconds=sim_t,robot_displacement_m=float(np.linalg.norm(pose[:2]-initial_pose[:2])),blocked_control_steps=blocked,target_speed_mps=.9,target_speed_semantics="nominal native gait; measured time-varying speed",target_actual_mean_speed_mps=target_path/sim_t,target_path_m=target_path,target_reached_route_end=stop_at is not None,min_target_clearance_m=min(x["clearance_m"] for x in gait),source=case,controller="LightNav body velocity -> official Spot policy 50Hz -> physical joint drives / PhysX500Hz; no base pose or velocity teleport; NavMesh audit only",embodiment=profile,target_motion="native motion matching with PathPoints; one initial placement only; Walk used only for start/stop blend")
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));print("SUMMARY",json.dumps(summary),flush=True)
app.close()
