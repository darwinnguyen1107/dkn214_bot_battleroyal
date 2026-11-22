"""Layered arena grid, cache and bounded bidirectional route planner."""
import hashlib
import heapq
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from array import array
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

CONFIG = {
    "plan_seconds": .018, "decision_seconds": .037, "cell": 3., "jump_height": 4.,
    "step": 1.2, "body_height": 9., "energy_floor": 30., "sound_memory": 4.,
    "track_memory": 3., "sector": 12, "burst_damage": 0.13, "log_limit": 2_000_000,
    "route_refresh_ticks": 28, "route_goal_shift": 9., "hold_seconds": .45,
    "damage_window": 2.5, "sustained_fraction": .04, "ambush_wait": .5,
    "farm_quiet": 5., "cover_peek": .25, "storm_trade_seconds": 2.,
    "max_sound_events": 48, "max_damage_events": 64, "max_gone": 4096,
    "map_seconds": 8., "terminal_circle": 45., "flank_angle": 2.094, "post_radii": (0, 9, 18, 30, 42)
}
DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

class Buckets:
    def __init__(self, items, width=CONFIG["sector"]):
        self.width = width
        self.cells = defaultdict(list)
        for item in items:
            x,y = item["pos"][:2]
            self.cells[math.floor(x/width), math.floor(y/width)].append(item)

    def near(self, p, r):
        x,y=p[:2]; w=self.width
        for i in range(math.floor((x-r)/w), math.floor((x+r)/w)+1):
            for j in range(math.floor((y-r)/w), math.floor((y+r)/w)+1):
                for item in self.cells.get((i,j), ()):
                    if dist(p,item["pos"]) <= r:
                        yield item

class LayeredMap:
    def __init__(self, map_dir):
        self.spans = {}
        self.learned = {}
        self.risk_cache = {}
        self.rebuild_context()
        root = Path(map_dir)/"v2"
        sig=hashlib.sha256(b"century-layers-v2"+sys.byteorder.encode())
        for name in ("world.json","terrain.i16"):
            with (root/name).open("rb") as stream:
                for chunk in iter(lambda:stream.read(262144),b""):
                    sig.update(chunk)
        cache_name="atlas-"+sig.hexdigest()+".bin"
        cache=Path(os.getenv("ATLAS_CACHE_DIR",tempfile.gettempdir()))/cache_name
        for source in (Path(__file__).with_name(cache_name),cache):
            try:
                with zipfile.ZipFile(source) as z:
                    meta=json.loads(z.read("meta"))
                    for key in ("tnx","tny","tminx","tminy","tstep","gnx","gny"):
                        setattr(self,key,meta[key])
                    self.terrain=array("h"); self.terrain.frombytes(z.read("terrain"))
                    self.top=array("f"); self.top.frombytes(z.read("top"))
                    self.wblock=bytearray(z.read("walls"))
                    self.spans={int(i):v for i,v in json.loads(z.read("levels")).items()}
                    if len(self.wblock)!=self.gnx*self.gny or len(self.top)!=len(self.wblock) or len(self.terrain)<self.tnx*self.tny:
                        raise ValueError("incomplete grid")
                return
            except (OSError,ValueError,KeyError,zipfile.BadZipFile):
                self.spans={}
        super().__init__(map_dir)
        temp=None
        try:
            with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as f:
                temp=f.name
            with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                z.writestr("meta",json.dumps({k:getattr(self,k) for k in ("tnx","tny","tminx","tminy","tstep","gnx","gny")}))
                z.writestr("terrain",self.terrain.tobytes());z.writestr("top",self.top.tobytes())
                z.writestr("walls",self.wblock);z.writestr("levels",json.dumps(self.spans))
            os.replace(temp,cache)
        except OSError:
            pass
        finally:
            if temp and os.path.exists(temp):
                os.unlink(temp)

    def rebuild_context(self,zone=None,hazards=(),enemies=(),body=9.):
        self.zone=zone
        self.hazards=hazards
        self.enemies=enemies
        self.body=body
        self.risk_cache.clear()

    def _apply(self,i,x,y,base,top):
        pair=(base,top)
        levels=self.spans.setdefault(i,[])
        if pair not in levels:
            levels.append(pair)
        super()._apply(i,x,y,base,top)

    def _inflate(self):
        if np is None:
            return super()._inflate()
        grid=np.frombuffer(self.wblock,dtype=np.uint8).reshape(self.gny,self.gnx)
        solid=np.pad(grid==1,1)
        margin=np.zeros(grid.shape,dtype=bool)
        for dx,dy in DIRS:
            margin |= solid[1+dy:1+dy+self.gny,1+dx:1+dx+self.gnx]
        grid[(grid==0)&margin]=2

    def floors(self,i):
        ground=self.ground(*self.center(i)); spans=self.spans.get(i,())
        return [z for z in sorted({ground,*(hi for _,hi in spans if hi>=ground)})
                if not any(lo < z+self.body-.1 and hi > z+.2 for lo,hi in spans)]

    def solid(self,p):
        i=self.cell_of(*p[:2])
        return (i is None or p[2]<self.ground(*p[:2]) or
                any(lo <= p[2] <= hi for lo,hi in self.spans.get(i,())))

    def shot_clear(self,a,b):
        steps=max(1,math.ceil(math.dist(tuple(a[:3]),tuple(b[:3]))/1.5))
        dx,dy,dz=(b[k]-a[k] for k in range(3))
        for k in range(steps+1):
            f=k/steps
            if self.solid((a[0]+dx*f,a[1]+dy*f,a[2]+dz*f)):return False
        return True

    def risk(self,node):
        if node in self.risk_cache:return self.risk_cache[node]
        i,z=node; x,y=self.center(i)
        value=sum(weight*max(0,1-dist((x,y),p)/r) for p,r,weight in self.hazards)
        for e in self.enemies[:2]:
            if dist((x,y),e["pos"])<140 and self.shot_clear((x,y,z+6),(*e["pos"][:2],e["pos"][2]+6)):
                value+=1.8
        self.risk_cache[node]=value
        return value

    def allowed(self,source,target):
        if not self.zone:return True
        for c,r in self.zone:
            # Inside routes stay inside; an outside spawn may move inward.
            if dist(target,c)>max(r,dist(source,c))+0.05:return False
        return True

    def neighbours(self,node,reverse):
        i,z=node;ix,iy=i%self.gnx,i//self.gnx
        p=self.center(i)
        for dx,dy in DIRS:
            nx,ny=ix+dx,iy+dy
            if not (0<=nx<self.gnx and 0<=ny<self.gny):continue
            j=ny*self.gnx+nx
            if j in self.learned:continue
            q=self.center(j)
            a,b=(q,p) if reverse else (p,q)
            if not self.allowed(a,b):continue
            for nz in self.floors(j):
                rise=z-nz if reverse else nz-z
                if not -7<=rise<=CONFIG["jump_height"]:continue
                if dx and dy:
                    sides=(iy*self.gnx+nx,ny*self.gnx+ix)
                    if any(k in self.learned or not any(abs(h-max(z,nz))<=1.2 for h in self.floors(k)) for k in sides):
                        continue
                jump=rise>CONFIG["step"]
                src_i,src_z=(j,nz) if reverse else (i,z)
                if jump and any(src_z<lo<src_z+self.body+rise for lo,_ in self.spans.get(src_i,())):continue
                dest=node if reverse else (j,nz)
                cost=(1.414 if dx and dy else 1.)*(1+self.risk(dest))
                cost+=3*jump+max(0,rise)*.22-min(.12,max(0,-rise)*.04)
                if self.wblock[dest[0]]==2:cost+=.8
                yield (j,nz),cost

    def route(self,pos,goal,deadline):
        s=self.cell_of(*pos[:2]); g=self.cell_of(*goal[:2])
        if s is None or g is None:return []
        sf=self.floors(s);gf=self.floors(g)
        if not sf or not gf:return []
        start=(s,min(sf,key=lambda z:abs(z-pos[2])))
        end=(g,min(gf,key=lambda z:abs(z-(goal[2] if len(goal)>2 else self.ground(*goal[:2])))))
        def heuristic(n,t):
            return dist(self.center(n[0]),self.center(t[0]))/self.CELL + .2*abs(n[1]-t[1])
        queues=[[(0,start)],[(0,end)]]; parents=[{},{}];costs=[{start:0},{end:0}]
        done=[set(),set()];meet=None;best=start
        side=0
        while queues[0] and queues[1] and time.perf_counter()<deadline:
            _,n=heapq.heappop(queues[side])
            if n not in done[side]:
                done[side].add(n)
                if n in costs[1-side]:
                    meet=n;break
                if side==0 and heuristic(n,end)<heuristic(best,end):best=n
                for nxt,edge in self.neighbours(n,side==1):
                    value=costs[side][n]+edge
                    if value<costs[side].get(nxt,float("inf")):
                        costs[side][nxt]=value;parents[side][nxt]=n
                        heapq.heappush(queues[side],(value+1.35*heuristic(nxt,start if side else end),nxt))
            side=1-side
        n=meet or best;nodes=[n]
        while n in parents[0]:
            n=parents[0][n];nodes.append(n)
        nodes.reverse()
        n=meet
        while n in parents[1]:
            n=parents[1][n];nodes.append(n)
        pts=[(*self.center(i),h) for i,h in nodes]
        # Corner cutting is permitted only on the same layer, in clear, safe cells.
        smoothed=[]
        for k,b in enumerate(pts):
            if 0<k<len(pts)-1 and time.perf_counter()<deadline and abs(pts[k-1][2]-b[2])+abs(pts[k+1][2]-b[2])<.6:
                a,c=pts[k-1],pts[k+1]
                u=tuple((a[j]+3*b[j])/4 for j in range(3))
                v=tuple((c[j]+3*b[j])/4 for j in range(3))
                if self.walk_clear(u,v) and self.allowed(u,v):
                    smoothed.extend((u,v));continue
            smoothed.append(b)
        return smoothed
    def _rasterize(self,prisms,t0):
        if np is None:return super()._rasterize(prisms,t0)
        def inside(x,y,ring):
            mask=np.zeros((len(y),x.size),dtype=bool)
            vertices=list(zip(ring[::2],ring[1::2]))
            for (x0,y0),(x1,y1) in zip(vertices,vertices[1:]+vertices[:1]):
                if y0!=y1:
                    mask ^= ((y0>y)!=(y1>y)) & (x < x0+(x1-x0)*(y-y0)/(y1-y0))
            return mask
        for prism in prisms:
            if time.perf_counter()-t0>CONFIG["map_seconds"]:raise TimeoutError("arena raster deadline")
            ring=prism.get("ring",[])
            if len(ring)<6:continue
            base,top=prism["base"],prism["top"]
            ix=np.arange(max(0,int((min(ring[::2])-self.tminx)/self.CELL)),min(self.gnx,int((max(ring[::2])-self.tminx)/self.CELL)+1))
            iy=np.arange(max(0,int((min(ring[1::2])-self.tminy)/self.CELL)),min(self.gny,int((max(ring[1::2])-self.tminy)/self.CELL)+1))
            x=self.tminx+(ix+.5)*self.CELL;y=self.tminy+(iy+.5)*self.CELL
            mask=inside(x,y[:,None],ring)
            holes=prism.get("holes") or []
            if holes and not isinstance(holes[0],list):holes=[holes]
            for hole in holes:mask &= ~inside(x,y[:,None],hole)
            rows,cols=np.nonzero(mask)
            for row,col in zip(rows,cols):
                self._apply(int(iy[row]*self.gnx+ix[col]),float(x[col]),float(y[row]),base,top)
            for px,py in zip(ring[::2],ring[1::2]):
                i=self.cell_of(px,py)
                if i is not None:self._apply(i,px,py,base,top)

    def walk_clear(self,a,b):
        length=dist(a,b)
        if np is None or length<24:return super().walk_clear(a,b)
        n=max(1,int(length/1.5))
        x=np.linspace(a[0],b[0],n+1);y=np.linspace(a[1],b[1],n+1)
        ix=np.floor((x-self.tminx)/self.CELL).astype(int)
        iy=np.floor((y-self.tminy)/self.CELL).astype(int)
        if np.any((ix<0)|(ix>=self.gnx)|(iy<0)|(iy>=self.gny)):return False
        ids=iy[1:]*self.gnx+ix[1:]
        if np.any(np.frombuffer(self.wblock,dtype=np.uint8)[ids]) or any(k in self.learned for k in ids):return False
        tx=np.clip(np.rint((x-self.tminx)/self.tstep).astype(int),0,self.tnx-1)
        ty=np.clip(np.rint((y-self.tminy)/self.tstep).astype(int),0,self.tny-1)
        h=np.frombuffer(self.terrain,dtype=np.int16)[ty*self.tnx+tx].astype(float)/100
        return bool(np.all(np.diff(h)<=max(1.2,length/n*1.19)))
