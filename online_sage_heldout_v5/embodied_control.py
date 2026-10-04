"""Habitat EVT Spot action adapter; displacement is preserved under SAGE's clock."""
import math,numpy as np
LIMITS=np.array([.375,.25,math.pi/20])
SCALES=np.array([15.,10.,6.28])
def waypoint_delta(waypoint):
 p=np.asarray(waypoint,float);assert p.shape==(3,) and np.isfinite(p).all()
 return np.clip(p/LIMITS,-1,1)*SCALES/40.
def action_target(pose,delta,fraction=1.):
 x,y,a=pose;c,s=math.cos(a),math.sin(a)
 dx,dy,da=np.asarray(delta)*fraction
 return np.array([x+c*dx-s*dy,y+s*dx+c*dy,math.atan2(math.sin(a+da),math.cos(a+da))])
def footprint(pose):
 p=np.asarray(pose);axis=np.array([math.cos(p[2]),math.sin(p[2])])
 return [p[:2]+d*axis for d in [0,.25,-.25]]
def project(nav,p,component=0):
 if nav.contains(p):return np.asarray(p[:2],float)
 tri=nav.tri[nav.components==component]
 a=tri.reshape(-1,2);b=np.roll(tri,-1,axis=1).reshape(-1,2);d=b-a
 t=np.clip(np.sum((p[:2]-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-20),0,1)
 q=a+t[:,None]*d
 return q[np.argmin(np.sum((q-p[:2])**2,axis=1))]
def navigate(nav,before,candidate,clearance_ok):
 start=footprint(before);goal=footprint(candidate)
 def valid(a,b):
  for p,q in zip(footprint(a),footprint(b)):
   for u in np.linspace(0,1,max(2,int(np.linalg.norm(q-p)/.01)+1)):
    x=p+u*(q-p)
    if not nav.contains(x) or not clearance_ok(x):return False
  return True
 if valid(before,candidate):return candidate,False,False
 # Habitat-style limiting-point sliding; closest boundary projection, never oracle steering.
 ends=[project(nav,q,nav.component(p)) for p,q in zip(start,goal)]
 idx=int(np.argmax([np.linalg.norm(q-e) for q,e in zip(goal,ends)]))
 sliding=candidate.copy();sliding[:2]=before[:2]+ends[idx]-start[idx]
 if np.linalg.norm(sliding[:2]-before[:2])<=max(np.linalg.norm(q-p) for p,q in zip(start,goal))+1e-6 and valid(before,sliding):
  return sliding,True,True
 # Conservative fallback also rejects rotations whose end footprint collides.
 return np.asarray(before).copy(),True,False
