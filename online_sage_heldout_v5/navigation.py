"""Queries against the independently baked and validated native NavMesh cache."""
import numpy as np,json,hashlib
class NativeNavigation:
 def __init__(self,root):
  self.root=root
  d=np.load(root/"native_nav_cache.npz");self.tri=d["triangles"][:,:,:2];self.components=d["components"]
  self.meta=json.loads((root/"native_nav_cache.json").read_text())
  assert abs(self.meta["baked_agent"]["radius"]-.3)<1e-5
  assert abs(self.meta["baked_agent"]["height"]-1.8)<1e-5
  self.a=self.tri[:,0];self.u=self.tri[:,1]-self.a;self.v=self.tri[:,2]-self.a
  self.det=self.u[:,0]*self.v[:,1]-self.u[:,1]*self.v[:,0]
 def component(self,p):
  w=np.asarray(p[:2])-self.a
  with np.errstate(divide="ignore",invalid="ignore"):
   s=(w[:,0]*self.v[:,1]-w[:,1]*self.v[:,0])/self.det
   t=(self.u[:,0]*w[:,1]-self.u[:,1]*w[:,0])/self.det
  ids=np.flatnonzero((abs(self.det)>1e-12)&(s>=-1e-6)&(t>=-1e-6)&(s+t<=1+1e-6))
  return int(self.components[ids[0]]) if len(ids) else -1
 def contains(self,p):return self.component(p)>=0
 def validate_case(self,case):
  points=[case["robot_start"]]+case["target_points"]
  components=[self.component(p) for p in points]
  assert min(components)>=0 and len(set(components))==1,components
  for a,b in zip(case["target_points"][:-1],case["target_points"][1:]):
   for p in np.linspace(a,b,max(2,int(np.linalg.norm(np.array(b)-a)/.02)+1)):
    assert self.contains(p),p
  return dict(radius_m=self.meta["baked_agent"]["radius"],height_m=self.meta["baked_agent"]["height"],triangles=len(self.tri),components=components,paths=case["target_points"],cache_sha256=hashlib.sha256((self.root/"native_nav_cache.npz").read_bytes()).hexdigest(),method="cached native NavMesh triangle queries; robot and target share connected component")
