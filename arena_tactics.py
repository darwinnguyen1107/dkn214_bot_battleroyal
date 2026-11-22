"""Decision extensions for the Century arena bot. All distances are game metres."""
import json
import math
import os
import tempfile
import time
from collections import deque, Counter
from pathlib import Path
from arena_nav import CONFIG, Buckets, dist

def heading(a,b):
    return math.atan2(b[1]-a[1],b[0]-a[0])

def delta(a,b):
    return (a-b+math.pi)%(2*math.pi)-math.pi

def step(p,a,r):
    return (p[0]+r*math.cos(a),p[1]+r*math.sin(a))

def clamp(v,a,b):
    return max(a,min(b,v))

class Decisions:
    def __init__(self, hello):
        super().__init__(hello)
        self.game=hello["config"]
        self.tracks={}
        self.ears=deque(maxlen=CONFIG["max_sound_events"])
        self.wounds=deque(maxlen=CONFIG["max_damage_events"])
        self.impacts=[]
        self.gone={}
        self.rotation_wait=None
        self.storm_goal=None
        self.retry={}
        self.last_seen=-100.
        self.state="POST"
        self.state_since=-100.
        self.pending_since={}
        self.deadline=0
        self.jump_link=False
        self.jump_dodge=False
        self.escape_tick=-100
        self.last_escape_tick=-200
        self.telemetry=Path(tempfile.gettempdir())/"century-trace.jsonl" if os.getenv("ATLAS_DEBUG")=="1" else None

    def log(self, record):
        if self.telemetry:
            try:
                if self.telemetry.exists() and self.telemetry.stat().st_size>=CONFIG["log_limit"]:return
                with self.telemetry.open("a",encoding="utf8") as out:
                    out.write(json.dumps(record,separators=(",",":"))+"\n")
            except OSError:
                self.telemetry=None

    def act(self,obs):
        start=time.perf_counter()
        self.deadline=start+CONFIG["decision_seconds"]
        self.obs=obs
        self.now=obs.get("t",obs["tick"]*.05)
        self.jump_link=self.jump_dodge=False
        self.visible=[];self.acoustic=[];self.impacts=[];self.collateral=set()
        self.objects=Buckets(());self.cubes=Buckets(())
        self.dps=0;self.alive=obs.get("alive",16)
        try:
            self.observe(obs)
        except (ValueError,KeyError,TypeError,ArithmeticError) as exc:
            self.log({"observe":str(exc)})
        action=super().act(obs)
        if self.telemetry:
            self.log({"t":self.now,"state":self.state,"ms":round(1000*(time.perf_counter()-start),2),"a":action})
        return action

    def observe(self,obs):
        me=obs["self"]; p=me["pos"]; now=self.now
        self.visible=[e for e in obs.get("mechs",()) if e["kind"]=="mech"]
        if self.visible:self.last_seen=now
        for e in self.visible:
            old=self.tracks.get(e["id"]); v=e.get("vel",(0,0,0))
            age=now-old[0] if old else 0
            accel=[clamp((a-b)/age,-24,24) for a,b in zip(v,old[1]["vel"])] if old and .02<age<.4 else [0,0,0]
            self.tracks[e["id"]]=(now,dict(e,vel=v),accel)
            if self.nav and abs(v[2])<1 and math.hypot(*v[:2])>1:
                self.nav.learned.pop(self.nav.cell_of(*e["pos"][:2]),None)
        self.tracks={i:v for i,v in self.tracks.items() if now-v[0]<CONFIG["track_memory"]}
        self.alive=obs.get("alive",16)
        self.cubes=Buckets(obs.get("cubes",()))
        self.objects=Buckets([*obs.get("crates",()),*obs.get("pickups",()),*obs.get("mechs",())])
        self.collateral=set()
        hazards=[]
        for beam in obs.get("beams",()):
            path=beam.get("path",())
            if len(path)<2:continue
            hazards.append((path[-1],12,4))
            for e in self.visible:
                if e["id"]!=beam.get("by") and dist(path[-1],e["pos"])<3 and dist(path[-1],p)>9:
                    self.collateral.add(e["id"])
        for evt in obs.get("events",()):
            if evt["type"]=="damaged":self.wounds.append((now,evt["amount"]))
            if evt["type"]=="take_failed":
                if evt.get("reason")=="gone":
                    self.gone[evt.get("id")]=True
                    if len(self.gone)>CONFIG["max_gone"]:self.gone.pop(next(iter(self.gone)))
                if evt.get("reason")=="range":self.retry[evt.get("id")]=now+2
        self.retry={i:t for i,t in self.retry.items() if now<t}
        self.dps=sum(d for t,d in self.wounds if now-t<CONFIG["damage_window"])/CONFIG["damage_window"]
        for snd in obs.get("sounds",()):
            kind=snd.get("kind"); a=snd.get("bearing")
            if a is None:continue
            age=snd.get("age",0);band=snd.get("dist")
            radius={"near":25,"mid":70,"far":140}.get(band,70)
            guess=step(p,a,radius); weight=1.
            for when,other,b,k,_,_ in reversed(self.ears):
                det=math.sin(b-a)
                if kind!=k or now-when>.8 or dist(p,other)<3 or abs(det)<.15:continue
                dx,dy=other[0]-p[0],other[1]-p[1]
                length=(dx*math.sin(b)-dy*math.cos(b))/det
                if radius*.4<length<radius*1.8:
                    point=step(p,a,length)
                    if abs(delta(heading(other,point),b))<.15:guess=point
                    break
            if self.nav and not self.nav.shot_clear((*p[:2],p[2]+6),(*guess,self.nav.ground(*guess)+6)):weight=.45
            self.ears.append((now-age,tuple(p),a,kind,guess,weight))
        self.acoustic=[(q,15,weight*(1-(now-t)/4)) for t,_,_,k,q,weight in self.ears if 0<=now-t<4 and k in ("gun","explosion","mortar","laser","lightning","step","jet","boost")]
        self.impacts=[]
        for shot in obs.get("projectiles",()):
            if shot["kind"]!="mortar":continue
            point,eta=self.impact(shot,p[2])
            if eta<=3:
                splash=self.game.get("weapons",{}).get("mortar",{}).get("splash_radius",7)
                self.impacts.append((point,eta,splash))
                hazards.append((point,splash+6,9/(1+eta)))
        self.storm_goal=None
        zone=obs["zone"];c=zone["center"];r=zone["radius"]
        # Catch the inside lip of a healing pad. Include capture and return in the HP budget.
        if me["hp"]<.6*me["max_hp"] and not self.visible and me["hp"]>zone.get("dps",20)*2:
            walk=max(1,me["stats"].get("walk_speed",11.7))
            for dr in obs.get("drops",()):
                if dr.get("holder") not in (None,me.get("id")) or dr.get("lands_in",0)>0:continue
                pad=step(dr["pos"],heading(dr["pos"],c),max(0,dr.get("radius",8)-1))
                remain=max(0,dr.get("capture_s",4)-(dr.get("progress",0) if dr.get("holder")==me.get("id") else 0))
                overshoot=dist(pad,c)-r
                if overshoot<=0 or dist(p,pad)>15:continue
                exposure=dist(p,pad)/walk+remain+overshoot/walk
                if exposure<=CONFIG["storm_trade_seconds"] and zone.get("dps",20)*exposure<min(.35*me["hp"],me["max_hp"]-me["hp"]) and (not self.nav or self.nav.walk_clear(p,pad)):
                    self.storm_goal=pad;break
        if self.nav:
            zone=obs["zone"]; c=zone["center"]; r=zone["radius"]
            nc=zone.get("next_center") or c
            nr=zone.get("next_radius") if zone.get("next_radius") is not None else r
            start=max(now,zone.get("shrink_start",now))
            frac=clamp((now+3-start)/max(.01,zone.get("shrink_end",start)-start),0,1)
            projected=[a+(b-a)*frac for a,b in zip(c,nc)]
            self.nav.rebuild_context([(c,r),(projected,r+(nr-r)*frac)],hazards+self.acoustic+[(e["pos"],18,3) for e in self.visible],sorted(self.visible,key=lambda e:dist(p,e["pos"])))
            if obs["tick"]%20<2:self.nav.expire(obs["tick"])

    def impact(self,shot,floor):
        p=shot["pos"];v=shot["vel"];g=self.mortar_g
        def position(t):
            return (p[0]+v[0]*t,p[1]+v[1]*t,p[2]+v[2]*t-g*t*t/2)
        def hit(t):
            q=position(t)
            return self.nav.solid(q) if self.nav else q[2]<=floor
        for i in range(1,49):
            end=i/12
            if hit(end):
                lo=end-1/12
                for _ in range(5):
                    mid=(lo+end)/2
                    if hit(mid):end=mid
                    else:lo=mid
                return position(end),end
        return position(4),4

    def steer(self,pos,goal,tick):
        a=heading(pos,goal); direct=(math.cos(a),math.sin(a)) if dist(pos,goal)>.5 else (0,0)
        if not self.nav:return direct
        try:
            stale=not self.path or self.path_goal is None or dist(goal,self.path_goal)>CONFIG["route_goal_shift"] or tick-self.plan_tick>CONFIG["route_refresh_ticks"]
            if stale and tick-self.plan_tick>=4 and time.perf_counter()<self.deadline:
                self.path=self.nav.route(pos,goal,min(self.deadline,time.perf_counter()+CONFIG["plan_seconds"]))
                self.pi=0;self.plan_tick=tick;self.path_goal=goal
            while self.pi<len(self.path):
                q=self.path[self.pi]
                if dist(pos,q)>2 or q[2]>pos[2]+1.2:
                    self.jump_link=q[2]>pos[2]+1.2 and dist(pos,q)<6
                    a=heading(pos,q)
                    return math.cos(a),math.sin(a)
                self.pi+=1
            self.path=[]
        except (ValueError,KeyError,TypeError,ArithmeticError,IndexError) as exc:self.log({"route":str(exc)})
        # Recover on the present layer if the search exhausted its time slice.
        cell=self.nav.cell_of(*pos[:2])
        if cell is not None:
            floors=self.nav.floors(cell)
            if floors:
                current=(cell,min(floors,key=lambda h:abs(h-pos[2])))
                if time.perf_counter()>=self.deadline:
                    a=heading(pos,goal);q=step(pos,a,4)
                    if abs(current[1]-self.nav.ground(*pos[:2]))<1 and self.nav.allowed(pos,q) and self.nav.walk_clear(pos,q):
                        return direct
                    return (0.,0.)
                candidates=[(edge+.9*dist(self.nav.center(n[0]),goal),n) for n,edge in self.nav.neighbours(current,False)]
                if candidates:
                    _,node=min(candidates);a=heading(pos,self.nav.center(node[0]))
                    self.jump_link=node[1]>pos[2]+1.2
                    return math.cos(a),math.sin(a)
        return (0.,0.)

    def lob(self,sx,sy,sz,target,vel,speed,grav):
        if min(speed,grav)<=0:return None
        try:
            for sign in (-1,1):
                flight=0
                for _ in range(5):
                    q=[p+v*flight for p,v in zip(target,vel)]
                    d=dist((sx,sy),q)
                    disc=speed**4-grav*(grav*d*d+2*(q[2]-sz)*speed**2)
                    if d<1 or disc<0:break
                    pitch=math.atan2(speed**2+sign*math.sqrt(disc),grav*d)
                    if not -.7<=pitch<=1.3:break
                    flight=d/(speed*math.cos(pitch))
                else:
                    yaw=heading((sx,sy),q); n=max(3,math.ceil(speed*flight/2))
                    previous=(sx,sy,sz)
                    for i in range(1,n+1):
                        t=flight*i/n
                        xy=step((sx,sy),yaw,speed*math.cos(pitch)*t)
                        point=(*xy,sz+speed*math.sin(pitch)*t-grav*t*t/2)
                        if self.nav and (self.nav.solid(point) or not self.nav.shot_clear(previous,point)):break
                        previous=point
                    else:return yaw,pitch,flight
        except (KeyError,ValueError,TypeError,ArithmeticError) as exc:self.log({"lob":str(exc)})
        return None

    def dodge(self,me,projectiles,center,safe_radius,rotation_options=None):
        try:
            p=me["pos"];walk=me.get("stats",{}).get("walk_speed",11.7)
            bullets=[]
            for shot in projectiles:
                if shot["kind"]=="mortar":continue
                s,v=shot["pos"],shot["vel"]
                # A candidate that sidesteps one projectile may enter another trajectory.
                if dist(p,s)<1.4*math.hypot(*v[:2])+24:bullets.append((s,v))
            blasts=[(q,t,r+2) for q,t,r in self.impacts if t<1.5 and dist(p,q)<r+22]
            if not bullets and not blasts:return None
            def danger(direction):
                vx,vy=(k*walk for k in direction)
                total=0.
                for s,v in bullets:
                    rx,ry=v[0]-vx,v[1]-vy
                    eta=clamp(((p[0]-s[0])*rx+(p[1]-s[1])*ry)/max(1,rx*rx+ry*ry),0,1.4)
                    z=s[2]+v[2]*eta-p[2]-me["vel"][2]*eta
                    if -1<z<10:
                        miss=math.hypot(s[0]-p[0]+rx*eta,s[1]-p[1]+ry*eta)
                        total+=max(0,5-miss)**2/(.2+eta)
                return total+sum(max(0,r-dist((p[0]+vx*t,p[1]+vy*t),q))**2/(.2+t) for q,t,r in blasts)
            if danger(tuple(me["vel"][k]/walk for k in (0,1)))==0:return None
            dirs=[(math.cos(i*math.pi/8),math.sin(i*math.pi/8)) for i in range(16)]
            if rotation_options:dirs=rotation_options(dirs)
            else:dirs=[v for v in dirs if dist((p[0]+8*v[0],p[1]+8*v[1]),center)<=max(safe_radius,dist(p,center))]
            if self.nav:dirs=[v for v in dirs if self.nav.walk_clear(p,(p[0]+8*v[0],p[1]+8*v[1]))]
            if not dirs:
                self.jump_dodge=True
                return None
            from arena_nav import np
            if np is not None and len(bullets)>3:
                vectors=np.asarray(dirs)*walk
                origin=np.asarray([s for s,_ in bullets])-np.asarray(p)
                speed=np.asarray([v for _,v in bullets])
                rel=speed[None,:,:2]-vectors[:,None,:]
                eta=np.clip(-(origin[None,:,:2]*rel).sum(axis=2)/np.maximum(1,(rel*rel).sum(axis=2)),0,1.4)
                miss=np.linalg.norm(origin[None,:,:2]+rel*eta[:,:,None],axis=2)
                height=origin[None,:,2]+(speed[None,:,2]-me["vel"][2])*eta
                loss=(np.maximum(0,5-miss)**2*((height>-1)&(height<10))/(.2+eta)).sum(axis=1)
                for q,t,r in blasts:
                    loss+=np.maximum(0,r-np.linalg.norm(np.asarray(p[:2])+vectors*t-np.asarray(q[:2]),axis=1))**2/(.2+t)
                return dirs[int(loss.argmin())]
            return min(dirs,key=danger)
        except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
            self.log({"dodge":str(exc)})
            return None

    def should_flee(self,me,enemy,enemies,hp_ratio,held):
        return (self.now<max(self.short_until,self.concuss_until) or self.dps>CONFIG["sustained_fraction"]*me["max_hp"] or
                self.alive==3 and hp_ratio<.62 or super().should_flee(me,enemy,enemies,hp_ratio,held))

    def covered(self,p):
        if not self.nav:return 0
        return sum(not self.nav.walk_clear(p,step(p,i*math.pi/4,8)) for i in range(8))

    def post_bonus(self,spot,eye,enemy,center,hurt):
        nav=self.nav;i=nav.cell_of(*spot)
        shelter=self.covered(spot)
        roof=i is not None and any(lo>eye[2] for lo,_ in nav.spans.get(i,()))
        high=clamp((eye[2]-7-enemy["pos"][2])/4,-1,2)
        body=any(entity.get("kind")=="npc" or "hp" in entity and "kind" not in entity
                 for entity in self.objects.near(spot,6) if abs(delta(heading(spot,entity["pos"]),heading(spot,enemy["pos"])))<.4)
        return high+.35*shelter+.8*(2<=shelter<=5)+1.4*roof+float(body)*.3-.008*dist(spot,center)

    def select_post(self,*args):
        if time.perf_counter()>=self.deadline:return self.post
        try:return super().select_post(*args)
        except (ValueError,TypeError,KeyError,ArithmeticError,IndexError) as exc:
            self.log({"post":str(exc)})
            return None

    def choose_upgrade(self,offer,hp_ratio=1.,has_secondary=False,bonuses=None):
        me=self.obs["self"];held={w["kind"] for w in me["weapons"] if w}
        stacks=Counter(u if isinstance(u,str) else u.get("id") for u in me.get("upgrades",()))
        lobby=Counter(k for _,e,_ in self.tracks.values() for k in e.get("weapons",()))
        late=self.alive<=4 or (self.obs["zone"].get("next_radius") or self.obs["zone"]["radius"])<45
        self.log({"t":self.now,"offer":offer.get("choices",[])})
        def score(c):
            k=c["id"];desc=(k+" "+c.get("desc","")).lower()
            base=(bonuses or {}).get(k,65)
            defense=any(v in desc for v in ("armor","plating","nanite","shield","max hp"))
            base+=(25 if late else 4)*defense
            base+=45*(hp_ratio<.42 and defense)
            base+=12*(not late and k in ("servos","gun_rate","gun_damage"))
            base+=8*((k=="nanites" and stacks["armor"]>0) or (k=="gun_mag" and stacks["gun_rate"]>0) or (k=="armor" and stacks["nanites"]>0))
            base+=8*(k=="servos" and lobby["mortar"]>lobby["laser"])
            base+=6*(defense and lobby["gun"]+lobby["laser"]>0)
            base+=12*("lightning" in held and ("energy" in desc or k=="lightning_efficiency"))
            base-=55*any(k.startswith(w+"_") and w not in held for w in ("mortar","laser","lightning","gun"))
            return base-6*stacks[k]-5*any(v in desc for v in ("penalty","slower","increased cost"))
        choices=offer.get("choices",[])
        return max(range(len(choices)),key=lambda i:score(choices[i])) if choices else None

    def scan(self,pos,c,t):
        a=heading(pos,c); suspects=[(time_,e) for time_,e,_ in self.tracks.values() if .05<t-time_<3]
        if suspects:
            when,e=max(suspects,key=lambda row:row[0]);age=t-when
            return heading(pos,[e["pos"][k]+age*e["vel"][k] for k in (0,1)])
        if self.acoustic:
            strengths=Counter()
            for q,_,weight in self.acoustic:
                sector=round(heading(pos,q)*12/(2*math.pi))%12
                strengths[sector]+=weight
            return strengths.most_common(1)[0][0]*(2*math.pi/12)
        offsets=(0,.5,-.5,1,-1)
        lanes=[a+v for v in offsets]
        if self.nav:
            lanes=[b for b in lanes if self.nav.shot_clear((*pos[:2],pos[2]+6),(*step(pos,b,32),pos[2]+6))] or lanes
        return lanes[int(t*1.5)%len(lanes)]

    def macro(self,goal,pursuing,urgent,enemy,hurt,posting):
        obs=self.obs;p=obs["self"]["pos"];z=obs["zone"];c=z.get("next_center") or z["center"]; r=z.get("next_radius") or z["radius"]
        proposed=("ROTATE" if urgent else "FLEE" if enemy and hurt else "POST" if posting else
                  "ENGAGE" if enemy and dist(p,enemy["pos"])<100 else
                  {"drop":"DROP","loot":"LOOT","salvage":"LOOT","farm":"FARM"}.get(pursuing,"POST"))
        if proposed!=self.state and (urgent or proposed=="FLEE" or self.now-self.state_since>CONFIG["hold_seconds"]):
            self.state=proposed;self.state_since=self.now
        if self.state=="FLEE" and self.post and (not self.nav or self.nav.allowed(p,self.post)):
            goal=self.post
        if self.state=="LOOT" and pursuing is None and not urgent and not enemy:
            local=[u["pos"] for u in self.cubes.near(p,12) if dist(u["pos"],c)<r and abs(u["pos"][2]-p[2])<3]
            if local:goal=min(local,key=lambda q:dist(p,q))
        if self.state=="DROP" and pursuing=="drop":
            pad=next((dr for dr in obs.get("drops",()) if dr["id"]==self.wanted_drop),None)
            if pad and pad.get("lands_in",0)<=0 and dist(p,pad["pos"])<pad.get("radius",8)-1:return None
        if self.state=="ENGAGE" and enemy and self.alive<=2 and goal is None:
            a=heading(enemy["pos"],p)
            goal=step(enemy["pos"],a,clamp(dist(p,enemy["pos"]),25,75))
        if not urgent and not hurt and enemy and dist(enemy["pos"],c)>r and dist(p,c)<r-5:
            angle=heading(c,enemy["pos"]);goal=step(c,angle,max(0,r-12))
        if self.state=="POST" and self.kill_site and self.now<self.aftermath_until and not pursuing and self.nav:
            choices=[step(p,self.kill_facing+a,12) for a in (1,-1,2,-2)]
            choices=[q for q in choices if dist(q,c)<r-4 and self.nav.walk_clear(p,q)]
            if choices:goal=max(choices,key=lambda q:self.covered(q)+.05*dist(q,self.kill_site))
        if self.state=="POST" and not enemy and not pursuing and not urgent and self.nav:
            for dr in obs.get("drops",()):
                holder=dr.get("holder")
                if holder is not None and holder!=obs["self"].get("id") and dist(p,dr["pos"])<80:
                    a=heading(dr["pos"],p)
                    spots=[step(dr["pos"],a+b,23) for b in (-.5,0,.5)]
                    spots=[q for q in spots if dist(q,c)<r and self.nav.walk_clear(p,q) and self.covered(q)]
                    if spots:goal=min(spots,key=lambda q:dist(p,q));break
        if self.storm_goal is not None:
            return self.storm_goal
        if not urgent and goal and not enemy and not pursuing and self.listen_bearing is not None:
            ahead=abs(delta(heading(p,goal),self.listen_bearing))<.4
            slack=z.get("shrink_start",self.now)-self.now-dist(p,goal)/max(1,obs["self"]["stats"].get("walk_speed",11.7))
            if ahead and slack>8 and obs["tick"]-self.listen_tick<12:
                if self.rotation_wait is None:self.rotation_wait=self.now
                if self.now-self.rotation_wait<1.5:return None
            else:self.rotation_wait=None
        if goal and not enemy and dist(p,goal)>8:
            a=heading(p,goal)
            cubes=[u["pos"] for u in self.cubes.near(p,12) if abs(u["pos"][2]-p[2])<3 and
                   abs(delta(heading(p,u["pos"]),a))<.4 and dist(u["pos"],goal)<dist(p,goal) and
                   (not urgent or dist(u["pos"],c)<dist(p,c)) and
                   (not self.nav or self.nav.allowed(p,u["pos"]) and self.nav.walk_clear(p,u["pos"]))]
            if cubes:goal=min(cubes,key=lambda q:dist(p,q))
        return goal
    def finish(self,obs,action,urgent,enemy,dodge,farm,goal,need,tts,escaping,clear,tgt):
        me=obs["self"];p=me["pos"];z=obs["zone"]; gun,second=me["weapons"]
        now=self.now;stats=me["stats"];energy=me["energy"]
        walk=stats.get("walk_speed",11.7);boost=stats.get("boost_speed",19.5)
        def safe(a,length=5):
            q=step(p,a,length)
            return (not self.nav or self.nav.allowed(p,q) and self.nav.walk_clear(p,q))
        def move(a):
            if safe(a):
                action["move"]=[math.cos(a),math.sin(a)]
                return True
            return False
        if urgent and not dodge and not escaping and me.get("in_zone",True):
            seconds=max(0,(need-max(0,tts))*walk/max(.1,boost-walk))
            action["boost"]=seconds>.1 and (tts<=seconds+1 or now>=z.get("shrink_start",now))
        if self.storm_goal is not None:
            a=heading(p,self.storm_goal)
            if dist(p,self.storm_goal)>1:action["move"]=[math.cos(a),math.sin(a)]
            else:action["move"]=[0.,0.]
        if escaping:
            if self.escape_tick!=self.escape_until:
                self.last_escape_tick,self.escape_tick=self.escape_tick,self.escape_until
            if self.escape_tick-self.last_escape_tick<=40 and energy>32:
                action["jump"]=bool(me.get("grounded",True))
                action["boost"]=True
        action["jump"]|=bool((self.jump_dodge or self.jump_link) and me.get("grounded",True))
        if enemy:
            a=heading(p,enemy["pos"]); d=dist(p,enemy["pos"])
            look=enemy.get("aim",[a+math.pi])[0]
            away=abs(delta(look,a))<.75
            if not urgent and not dodge and not escaping:
                others=[heading(p,e["pos"]) for e in self.visible if e["id"]!=enemy["id"] and dist(p,e["pos"])<100]
                pinch=any(abs(delta(b,a))>CONFIG["flank_angle"] for b in others) and d<100
                charged=enemy.get("charging") and abs(delta(look,a+math.pi))<.20
                if pinch or charged:
                    options=[a+math.pi/2,a-math.pi/2]
                    def protection(b):
                        q=step(p,b,6)
                        return 10*(self.nav is not None and not self.nav.shot_clear((*q,p[2]+6),(*enemy["pos"][:2],enemy["pos"][2]+6)))+min(dist(q,e["pos"]) for e in self.visible)/20
                    for b in sorted(options,key=protection,reverse=True):
                        if move(b):
                            action["boost"]=charged or pinch
                            break
                    else:
                        # Consider a brief border crossing only when both safe lanes close.
                        if pinch and self.nav:
                            for b in sorted(options,key=protection,reverse=True):
                                q=step(p,b,5);outside=dist(q,z["center"])-z["radius"]
                                seconds=(5+max(0,outside))/walk
                                if 0<outside and seconds<=2 and seconds*z.get("dps",20)<min(me["hp"]*.15,self.dps*2) and self.nav.walk_clear(p,q):
                                    action["move"]=[math.cos(b),math.sin(b)];action["boost"]=True;break
                elif dist(enemy["pos"],z["center"])>z["radius"] and goal and me.get("in_zone",True) and me["hp"]>.65*me["max_hp"]:
                    if dist(p,goal)>4:move(heading(p,goal))
                elif self.state=="FLEE" and self.post:
                    move(heading(p,self.post))
                elif now-self.reload_time<.75 and self.reload_sound is not None and abs(delta(a,self.reload_sound))<.25 and d>35 and self.dps<1:
                    move(a+(.45 if self.strafe_sign>0 else -.45))
                elif self.state=="POST" and self.nav and self.post and dist(p,self.post)<5:
                    hide=gun and gun.get("reload",0)>0 or second and second.get("cooldown",0)>.3
                    for b in (a+math.pi/2,a-math.pi/2):
                        q=step(p,b,walk*CONFIG["cover_peek"])
                        if self.nav.shot_clear((*q,p[2]+6),(*enemy["pos"][:2],enemy["pos"][2]+5))!=bool(hide) and dist(q,self.post)<5 and move(b):break
                if away and self.state!="FLEE" and not pinch and not charged and self.dps==0 and d<80:
                    since=self.pending_since.setdefault(enemy["id"],now)
                    if now-since<CONFIG["ambush_wait"]:
                        action["move"]=[v*min(1,.85*self.step_speed/walk) for v in action["move"]]
                        action["boost"]=False;action["fire"][0]=False
                        if second and second["kind"]=="laser":action["fire"][1]=True
                        else:action["fire"][1]=False
                else:self.pending_since.pop(enemy["id"],None)
            if self.state=="FLEE" and self.damage_bearing is not None and now-self.last_damage_t<.8 and not any(action["fire"]):
                action["aim"]=[self.damage_bearing,0.]
        if second and second["kind"]=="laser" and enemy and not clear and not urgent and self.state=="POST" and now>=self.short_until:
            charge=second.get("charge",0)
            action["fire"][1]=second.get("cooldown",0)<=0 and charge<second["stats"].get("charge",.75) and self.post is not None and dist(p,self.post)<5
        if farm and farm[1].get("kind")=="npc" and dist(p,farm[1]["pos"])<34 and not enemy and not dodge and not urgent:
            a=heading(farm[1]["pos"],p)
            for b in (a+.45*self.strafe_sign,a,a-.45*self.strafe_sign):
                if move(b):break
        if gun:
            reload=gun.get("reload",0)
            if not action["fire"][0] and reload<=0 and gun.get("ammo",0)<=gun["stats"].get("mag",12)//2 and (action["jump"] or action["boost"] or not clear) and (not enemy or dist(p,enemy["pos"])>40):
                action["reload"]=True
            if 0<reload<.45 and (not enemy or dist(p,enemy["pos"])>16):
                action["fire"][0]=False
                if action.get("drop")==0:action.pop("drop",None)
        if self.telemetry and self.impacts:
            action["debug"]={"points":[{"at":list(q),"size":6,"color":"#ff8040"} for q,_,_ in self.impacts[:4]]}
        if not tgt and now-self.jet_time<.6:action.setdefault("aim",list(me["aim"]))[1]=.13
        if energy<CONFIG["energy_floor"]+stats.get("boost_drain",14)*.1 and not (urgent or dodge or self.jump_dodge):
            action["boost"]=False;action["jump"]=False
        if second and second["kind"]=="lightning" and energy-second["stats"].get("energy",3)<CONFIG["energy_floor"]:
            action["fire"][1]=False
        self.pending_since={i:v for i,v in self.pending_since.items() if i in self.tracks}
        return action
