"""Online control math. Coordinates: x forward, y left, yaw counterclockwise."""
import math
import numpy as np

def waypoint_to_command(waypoint, dt=.25, max_v=1.5, max_w=2.):
    w=np.asarray(waypoint,dtype=float)
    if w.shape!=(3,) or not np.isfinite(w).all() or dt<=0:
        raise ValueError("Invalid waypoint or controller period")
    # Same differential command mapping as LightNav first_waypoint_to_velocity_cmd.
    return float(np.clip(w[0]/dt,0,max_v)),float(np.clip(w[2]/dt,-max_w,max_w))

def integrate(pose,v,w,dt):
    x,y,yaw=map(float,pose)
    if abs(w)<1e-8: x+=v*dt*math.cos(yaw); y+=v*dt*math.sin(yaw)
    else:
        x+=v/w*(math.sin(yaw+w*dt)-math.sin(yaw))
        y-=v/w*(math.cos(yaw+w*dt)-math.cos(yaw))
    yaw=math.atan2(math.sin(yaw+w*dt),math.cos(yaw+w*dt))
    return np.asarray([x,y,yaw])

def path_at_distance(points,distance):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError("Invalid target path")
    delta=np.diff(p[:,:2],axis=0); lengths=np.linalg.norm(delta,axis=1)
    valid=lengths>1e-6
    if not valid.any(): raise ValueError("Stationary target path")
    keep=np.r_[True,valid]; p=p[keep]; lengths=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
    s=np.r_[0,np.cumsum(lengths)]; d=float(np.clip(distance,0,s[-1])); j=min(np.searchsorted(s,d,side="right")-1,len(p)-2)
    t=(d-s[j])/lengths[j]; pos=p[j]*(1-t)+p[j+1]*t
    yaw=math.atan2(p[j+1,1]-p[j,1],p[j+1,0]-p[j,0])
    return pos,yaw,d>=s[-1]
