"""Darwin Survivor v2: placement-focused Mech Royale bot with map-aware navigation."""
import heapq
import math
import os
import random
import sys
import time
from array import array
from collections import deque

import mechfight as mf
from arena_nav import LayeredMap, CONFIG
from arena_tactics import Decisions

# Feature flags. Flip ONE at a time between submissions to see what helps.
USE_NAV = os.getenv("ATLAS_USE_NAV", "1") == "1"    # A* over the campus map instead of walking straight lines
SETTLE = os.getenv("ATLAS_SETTLE", "1") == "1"     # wait on a ring inside the next zone, not at its exact center
STRAFE = os.getenv("ATLAS_STRAFE", "1") == "1"     # keep moving sideways while fighting
FARM = os.getenv("ATLAS_FARM", "1") == "1"       # shoot nearby crates / NPC drones for cubes when it is quiet
DROPS = os.getenv("ATLAS_DROPS", "1") == "1"      # go for supply drops when it looks safe
PLAN_BUDGET_S = CONFIG["plan_seconds"]   # max wall time for one A* search (tick budget is 50 ms)
MAP_BUDGET_S = CONFIG["map_seconds"]      # max time spent building the nav grid at startup

BEATS = {"laser": "mortar", "mortar": "lightning", "lightning": "laser"}
NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(dx, dy):
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 0.001 else (0.0, 0.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def point_in_poly(px, py, ring):
    inside = False
    n = len(ring) // 2
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class BaseGrid:
    """Coarse walkability grid built from v2/world.json prisms and terrain."""
    CELL = CONFIG["cell"]

    def __init__(self, map_dir):
        import json
        t0 = time.perf_counter()
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            world = json.load(f)
        tr = world["terrain"]
        self.tminx = float(tr["minX"])
        self.tminy = float(tr["minY"])
        self.tstep = float(tr.get("step", 1.0))
        self.tnx = int(tr["nx"])
        self.tny = int(tr["ny"])
        raw = array("h")
        with open(os.path.join(map_dir, "v2", "terrain.i16"), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder == "big":
            raw.byteswap()
        if len(raw) < self.tnx * self.tny:
            raise ValueError("terrain file too small")
        self.terrain = raw
        self.gnx = int(math.ceil(self.tnx * self.tstep / self.CELL))
        self.gny = int(math.ceil(self.tny * self.tstep / self.CELL))
        n = self.gnx * self.gny
        self.top = array("f", [-1e9]) * n     # tallest solid in the cell (absolute z)
        self.wblock = bytearray(n)            # 1 = solid, 2 = next to solid (costly)
        self.learned = {}                     # cell -> tick it expires (found by getting stuck)
        self._rasterize(world.get("prisms", []), t0)
        del world
        self._inflate()

    def ground(self, x, y):
        i = min(max(int(round((x - self.tminx) / self.tstep)), 0), self.tnx - 1)
        j = min(max(int(round((y - self.tminy) / self.tstep)), 0), self.tny - 1)
        return self.terrain[j * self.tnx + i] / 100.0

    def cell_of(self, x, y):
        cx = int((x - self.tminx) / self.CELL)
        cy = int((y - self.tminy) / self.CELL)
        if x < self.tminx or y < self.tminy or cx >= self.gnx or cy >= self.gny:
            return None
        return cy * self.gnx + cx

    def center(self, idx):
        cx, cy = idx % self.gnx, idx // self.gnx
        return (self.tminx + (cx + 0.5) * self.CELL, self.tminy + (cy + 0.5) * self.CELL)

    def _apply(self, idx, gx, gy, base, top):
        g = self.ground(gx, gy)
        if base < g + 9.0:                    # something a mech body would hit
            if top > self.top[idx]:
                self.top[idx] = top
            if top > g + 1.2:                 # taller than a step: a wall
                self.wblock[idx] = 1

    def _rasterize(self, prisms, t0):
        C = self.CELL
        for p in prisms:
            if time.perf_counter() - t0 > MAP_BUDGET_S:
                raise TimeoutError("map too slow to rasterize")
            try:
                ring = p["ring"]
                base = float(p["base"])
                top = float(p["top"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(ring) < 6:
                continue
            holes = p.get("holes") or []
            if holes and not isinstance(holes[0], (list, tuple)):
                holes = [holes]
            xs, ys = ring[0::2], ring[1::2]
            cx0 = max(0, int((min(xs) - self.tminx) / C))
            cx1 = min(self.gnx - 1, int((max(xs) - self.tminx) / C))
            cy0 = max(0, int((min(ys) - self.tminy) / C))
            cy1 = min(self.gny - 1, int((max(ys) - self.tminy) / C))
            for cy in range(cy0, cy1 + 1):
                py = self.tminy + (cy + 0.5) * C
                for cx in range(cx0, cx1 + 1):
                    px = self.tminx + (cx + 0.5) * C
                    if not point_in_poly(px, py, ring):
                        continue
                    if holes and any(point_in_poly(px, py, h) for h in holes):
                        continue
                    self._apply(cy * self.gnx + cx, px, py, base, top)
            for k in range(0, len(ring) - 1, 2):     # thin walls between cell centers
                idx = self.cell_of(ring[k], ring[k + 1])
                if idx is not None:
                    self._apply(idx, ring[k], ring[k + 1], base, top)

    def _inflate(self):
        w, gnx, gny = self.wblock, self.gnx, self.gny
        src = bytes(w)
        for cy in range(gny):
            row = cy * gnx
            for cx in range(gnx):
                if src[row + cx] != 1:
                    continue
                for dy in (-1, 0, 1):
                    yy = cy + dy
                    if yy < 0 or yy >= gny:
                        continue
                    for dx in (-1, 0, 1):
                        xx = cx + dx
                        if 0 <= xx < gnx and w[yy * gnx + xx] == 0:
                            w[yy * gnx + xx] = 2

    def expire(self, tick):
        for k in [k for k, v in self.learned.items() if v < tick]:
            del self.learned[k]

    def learn_block(self, x, y, d, tick):
        for k in (3.0, 6.0):
            idx = self.cell_of(x + d[0] * k, y + d[1] * k)
            if idx is not None and (idx in self.learned or len(self.learned) < 3000):
                self.learned[idx] = tick + (900 if idx in self.learned else 300)

    def walk_clear(self, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        previous_ground = self.ground(*a[:2])
        n = max(1, int(math.hypot(dx, dy) / 1.5))
        for k in range(1, n + 1):
            f = k / n
            px, py = a[0]+dx*f, a[1]+dy*f
            ground = self.ground(px,py)
            if ground - previous_ground > max(1.2, math.hypot(dx,dy)/n * 1.19):
                return False
            previous_ground = ground
            idx = self.cell_of(px,py)
            if idx is None or self.wblock[idx] or idx in self.learned:
                return False
        return True

    def shot_clear(self, a, b):
        """a, b = (x, y, z). False if a building clearly sits in the way."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = int(math.hypot(dx, dy) / 1.5)
        run = 0
        for k in range(2, n - 1):
            f = k / n
            idx = self.cell_of(a[0] + dx * f, a[1] + dy * f)
            line_z = a[2] + dz * f
            if self.ground(a[0]+dx*f, a[1]+dy*f) > line_z + 0.3:
                return False
            if idx is not None and self.top[idx] > line_z + 0.5:
                run += 1
                if run >= 2:
                    return False
            else:
                run = 0
        return True

    def nearest_open(self, idx):
        if self.wblock[idx] != 1 and idx not in self.learned:
            return idx
        gnx = self.gnx
        cx, cy = idx % gnx, idx // gnx
        for r in range(1, 9):
            best, best_d = None, 1e9
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    xx, yy = cx + dx, cy + dy
                    if not (0 <= xx < gnx and 0 <= yy < self.gny):
                        continue
                    j = yy * gnx + xx
                    if self.wblock[j] == 1 or j in self.learned:
                        continue
                    if dx * dx + dy * dy < best_d:
                        best, best_d = j, dx * dx + dy * dy
            if best is not None:
                return best
        return None

    def plan(self, start, goal, budget_s):
        s = self.cell_of(*start)
        g = self.cell_of(*goal)
        if s is None or g is None:
            return None
        s, g = self.nearest_open(s), self.nearest_open(g)
        if s is None or g is None:
            return None
        gnx, gny = self.gnx, self.gny
        gcx, gcy = g % gnx, g // gnx
        wblock, learned = self.wblock, self.learned

        def h(i):
            dx, dy = abs(i % gnx - gcx), abs(i // gnx - gcy)
            return 1.3 * (max(dx, dy) + 0.414 * min(dx, dy))

        openh = [(h(s), s)]
        gs = {s: 0.0}
        came = {}
        closed = set()
        best, besth = s, h(s)
        t0 = time.perf_counter()
        n = 0
        while openh:
            _, i = heapq.heappop(openh)
            if i in closed:
                continue
            closed.add(i)
            if i == g:
                best = g
                break
            hi = h(i)
            if hi < besth:
                best, besth = i, hi
            n += 1
            if (n & 15) == 0 and time.perf_counter() - t0 > budget_s:
                break
            cx, cy = i % gnx, i // gnx
            gi = gs[i]
            for dx, dy, c in NEIGH:
                xx, yy = cx + dx, cy + dy
                if not (0 <= xx < gnx and 0 <= yy < gny):
                    continue
                j = yy * gnx + xx
                if j in closed:
                    continue
                w = wblock[j]
                if w == 1 or j in learned:
                    continue
                if dx and dy and any(wblock[side] == 1 or side in learned
                                     for side in (cy*gnx+xx, yy*gnx+cx)):
                    continue
                rise = self.ground(*self.center(j)) - self.ground(*self.center(i))
                if rise > self.CELL*c*1.19:
                    continue
                ng = gi + c * (4.0 if w == 2 else 1.0) + 0.12 * max(0,rise)
                if ng < gs.get(j, 1e18):
                    gs[j] = ng
                    came[j] = i
                    heapq.heappush(openh, (ng + h(j), j))
        cells = [best]
        while cells[-1] in came:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [self.center(i) for i in cells]
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            k = i + 1
            if time.perf_counter()-t0 >= budget_s:
                return out + pts[i+1:]
            while k + 1 < len(pts) and time.perf_counter()-t0 < budget_s and self.walk_clear(pts[i], pts[k + 1]):
                k += 1
            out.append(pts[k])
            i = k
        return out


class NavGrid(LayeredMap, BaseGrid):
    pass


class CenturyBot:
    name = "Darwin Century"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.my_slot = hello.get("slot", 0)
        self.rng = random.Random(hello.get("seed", 0))
        self.pickup_radius = cfg["loot"]["pickup_radius"]
        self.cube_radius = cfg.get("cubes", {}).get("pickup_radius", 3.5)
        self.nav = None
        if USE_NAV:
            try:
                self.nav = NavGrid(hello["map"]["dir"])
            except Exception as exc:  # no map is better than no bot
                print("nav disabled: %r" % (exc,), file=sys.stderr)
        self.hist = deque(maxlen=10)
        self.moving_obs = 0
        self.escape_until = -1
        self.escape_dir = (0.0, 0.0)
        self.path = []
        self.pi = 0
        self.path_goal = None
        self.plan_tick = -999
        self.recent_contacts = {}
        self.kill_site = None
        self.aftermath_until = -1.0
        self.last_hit_id = None
        self.last_hit_time = -100.
        self.rear_time = -100.
        self.failed_pickups = {}
        self.reload_sound = self.jet_sound = None
        self.reload_time = self.jet_time = -100.
        self.zone_key = None
        self.recovering = False
        self.post = self.post_key = None
        self.post_tick = -100
        self.short_until = self.concuss_until = self.swap_until = -100.0
        self.landed_kills = 0
        self.damage_burst = 0
        self.last_damage_t = -100.
        self.kill_facing = 0.0
        self.wanted_drop = None
        self.focus_id = None
        self.attacker = None
        self.step_bearing = None
        self.step_time = -100.0
        self.mortar_g = cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
        self.step_speed = cfg.get("sound", {}).get("step_speed", 5.0)
        self.upgrade_thresholds = cfg.get("cubes", {}).get("upgrade_thresholds", [1, 3, 6, 10, 15, 21])
        self.damage_bearing = None
        self.damage_tick = -30
        self.listen_bearing = None
        self.listen_tick = -30
        self.strafe_sign = 1
        self.strafe_until = 0

    # ------------------------------------------------------------------ helpers
    def choose_upgrade(self, offer, hp_ratio=1.0, has_secondary=False, bonuses=None):
        choices = offer.get("choices", [])
        terms = [("armor", 100), ("repair", 95), ("regen", 90), ("health", 85), ("hp", 85),
                 ("speed", 70), ("energy", 65), ("shield", 60), ("sensor", 55)]

        preferred = {"armor": 101, "plating": 98, "nanites": 104 if hp_ratio < 0.65 else 89,
                     "gun_damage": 95 if not has_secondary else 81, "gun_rate": 80,
                     "gun_mag": 79, "laser_damage": 84, "lightning_range": 84}

        def priority(c):
            if bonuses and c.get("id") in bonuses:
                return bonuses[c["id"]]
            if c.get("id") in preferred:
                return preferred[c["id"]]
            s = (c.get("id", "") + " " + c.get("name", "") + " " + c.get("desc", "")).lower()
            return max([sc for word, sc in terms if word in s] or [10])
        return max(range(len(choices)), key=lambda i: priority(choices[i])) if choices else None

    def dodge(self, me, projectiles, center, safe_radius, rotation_options=None):
        """Sidestep the earliest projectile predicted to pass near the mech."""
        x, y, z = me["pos"]
        mx, my, mz = me["vel"]
        earliest = 1.25
        direction = None
        for shot in projectiles:
            px, py, pz = shot["pos"]
            vx, vy, vz = shot["vel"]
            vx, vy = vx - mx, vy - my
            speed_sq = vx * vx + vy * vy
            if speed_sq < 1:
                continue
            t = ((x - px) * vx + (y - py) * vy) / speed_sq
            if not 0 < t <= earliest:
                continue
            dx, dy = px - x + vx * t, py - y + vy * t
            margin = 7.5 if shot["kind"] == "mortar" else 4.5
            height_error = pz - z - 4.5 + (vz - mz) * t
            if shot["kind"] == "mortar":
                height_error -= 0.5 * self.mortar_g * t * t
            if math.hypot(dx, dy) >= margin or abs(height_error) > margin:
                continue
            sx, sy = unit(-vy, vx)
            options = [(sx, sy), (-sx, -sy)]
            if rotation_options is None:
                safe = [v for v in options if distance((x + 10 * v[0], y + 10 * v[1]), center) <= safe_radius]
            else:
                safe = rotation_options(options)
            if self.nav and rotation_options is None:
                safe = [v for v in safe if self.nav.walk_clear((x,y), (x+10*v[0], y+10*v[1]))]
            if safe:
                direction = max(safe, key=lambda v: (
                    -(dx * v[0] + dy * v[1]) + 0.2*(mx*v[0]+my*v[1]), -distance((x + 10 * v[0], y + 10 * v[1]), center)))
                earliest = t
        return direction

    def steer(self, pos, goal, tick):
        """Unit vector toward goal, following an A* path around buildings when possible."""
        x, y = pos[0], pos[1]
        direct = unit(goal[0] - x, goal[1] - y)
        nav = self.nav
        if nav is None:
            return direct
        try:
            if pos[2] > nav.ground(x, y) + 3.0:          # roof / bridge / airborne
                return direct
            if distance((x, y), goal) < 45 and nav.walk_clear((x, y), goal):
                return direct
            stale = (not self.path or self.path_goal is None
                     or distance(self.path_goal, goal) > 15 or tick - self.plan_tick > 40)
            if stale and tick - self.plan_tick >= 4:
                nav.expire(tick)
                self.plan_tick = tick
                self.path_goal = goal
                self.path = nav.plan((x, y), goal, PLAN_BUDGET_S) or []
                self.pi = 0
            p = self.path
            if not p:
                return direct
            while self.pi < len(p) - 1 and distance((x, y), p[self.pi]) < 4.5:
                self.pi += 1
            j = self.pi
            while j + 1 < len(p) and distance((x, y), p[j + 1]) < 40 and nav.walk_clear((x, y), p[j + 1]):
                j += 1
            self.pi = j
            if distance((x, y), p[j]) < 1.0:
                if j == len(p)-1 and distance(p[j], goal)>5:
                    self.path = []
                    self.plan_tick = tick-4
                return direct
            return unit(p[j][0] - x, p[j][1] - y)
        except Exception:
            return direct

    def lob(self, sx, sy, sz, target, vel, speed, grav):
        """Low-angle ballistic solution with target lead. Returns (yaw, pitch, flight_time) or None."""
        tx, ty, tz = target
        T = 0.0
        yaw = pitch = 0.0
        v2 = speed * speed
        for _ in range(4):
            px, py = tx + vel[0] * T, ty + vel[1] * T
            d = math.hypot(px - sx, py - sy)
            dz = tz - sz
            disc = v2 * v2 - grav * (grav * d * d + 2.0 * dz * v2)
            if disc < 0 or d < 1.0:
                return None
            pitch = math.atan2(v2 - math.sqrt(disc), grav * d)
            T = min(3.5, d / (speed * math.cos(pitch)))
            yaw = math.atan2(py - sy, px - sx)
        return yaw, pitch, T

    def select_post(self, pos, enemy, center, limit, hurt, enemies, incoming):
        nav = self.nav
        x,y,z = pos
        ex,ey,ez = enemy["pos"]
        best, best_score = None, 0
        for radius in CONFIG['post_radii']:
            if time.perf_counter() >= self.deadline:break
            for dx,dy,norm in (NEIGH if radius else [(0,0,1)]):
                if time.perf_counter()>=self.deadline:return best
                spot = (x+radius*dx/norm,y+radius*dy/norm)
                if distance(spot,center)>limit or (radius and not nav.walk_clear(pos,spot)):
                    continue
                eye = (*spot,nav.ground(*spot)+7)
                clear = nav.shot_clear(eye,(ex,ey,ez+4.5))
                if hurt:
                    score = 8 * (not clear)
                else:
                    if not clear:
                        continue
                    back = math.atan2(spot[1]-ey,spot[0]-ex)
                    shield = 0
                    for offset in (-0.5,0,0.5):
                        px = spot[0]+20*math.cos(back+offset)
                        py = spot[1]+20*math.sin(back+offset)
                        shield += not nav.shot_clear(eye,(px,py,nav.ground(px,py)+7))
                    score = shield if shield >= 2 else 0
                score -= radius/25
                if score > 0:
                    if spot == self.post:
                        score += 0.6
                    wall = nav.cell_of(spot[0]+8*dx/norm, spot[1]+8*dy/norm)
                    if wall is not None and nav.top[wall] > eye[2]+12:
                        score += 0.2
                    escape_x,escape_y = unit(center[0]-spot[0],center[1]-spot[1])
                    if nav.walk_clear(spot,(spot[0]+6*escape_x,spot[1]+6*escape_y)):
                        score += 0.4
                    for other in enemies[1:3]:
                        ox,oy,oz = other["pos"]
                        if nav.shot_clear(eye,(ox,oy,oz+7)):
                            score -= 1
                    if any(distance(spot,q["pos"]) < 10 for q in incoming if q["kind"]=="mortar"):
                        continue
                    # Avoid moving toward a second hunter to reach cover.
                    if any(distance(spot,e["pos"])+8 < distance(pos,e["pos"])
                           and distance(spot,e["pos"])<40 for e in enemies[1:]):
                        continue
                if hurt or clear:
                    score += self.post_bonus(spot,eye,enemy,center,hurt)
                if score > best_score:
                    best,best_score = spot,score
        return best

    def should_flee(self, me, enemy, enemies, hp_ratio, held):
        max_hp = max(1, me["max_hp"])
        e_hp = enemy.get("hp")
        if e_hp is None:
            e_hp = 0.8 * enemy.get("max_hp", max_hp)
        edge = (me["hp"] - e_hp) / max_hp + 0.04 * (me.get("cubes", 0) - enemy.get("cubes", 0))
        theirs = set(enemy.get("weapons", []))
        for k in held:
            if BEATS.get(k) in theirs:
                edge += 0.12
        for k in theirs:
            if BEATS.get(k) in held:
                edge -= 0.12
        others = sum(1 for m in enemies[1:] if distance(me["pos"], m["pos"]) < 90)
        edge -= 0.25 * others
        if enemy.get("cubes",0)>=me.get("cubes",0)+5 and self.alive>3:edge-=.25
        return edge < -0.2 or hp_ratio < 0.35

    # --------------------------------------------------------------------- act
    def act(self, obs: mf.Observation) -> mf.Action:
        try:
            return self._act(obs)
        except Exception as exc:   # an exception would idle the mech for the whole match
            print("act error: %r" % (exc,), file=sys.stderr)
            return self._fallback(obs)

    def _fallback(self, obs):
        action = {"tick": obs["tick"], "fire": [False, False]}
        try:
            me, zone = obs["self"], obs["zone"]
            c = zone.get("next_center") or zone["center"]
            action["move"] = list(unit(c[0] - me["pos"][0], c[1] - me["pos"][1]))
        except Exception:
            pass
        return action

    def _act(self, obs):
        me = obs["self"]
        pos = me["pos"]
        x, y, z = pos[0], pos[1], pos[2]
        tick = obs["tick"]
        t = obs.get("t", tick * 0.05)
        zone = obs["zone"]
        center = zone["center"]
        radius = zone["radius"]
        next_center = zone.get("next_center") or center
        next_radius = zone.get("next_radius")
        if next_radius is None:
            next_radius = radius
        hp_ratio = me["hp"] / max(1, me["max_hp"])
        energy = me["energy"]
        weapons = me["weapons"]
        stats = me.get("stats", {})
        walk = max(1.0, stats.get("walk_speed", 11.7))
        muzzle_z = z + stats.get("muzzle_height", 7.0)
        held = {w["kind"] for w in weapons if w}
        my_id = me.get("id", self.my_slot)

        mechs = obs.get("mechs", [])
        enemies = sorted((m for m in mechs if m["kind"] == "mech"), key=lambda m: distance(pos, m["pos"]))
        npcs = [m for m in mechs if m["kind"] == "npc"]
        enemy = enemies[0] if enemies else None
        enemy_dist = distance(pos, enemy["pos"]) if enemy else float("inf")
        action = {"tick": tick, "fire": [False, False]}

        # The kill event carries an ID. Keep a short observation history so
        # the location is available even if the mech has already disappeared.
        for e in enemies:
            self.recent_contacts[e["id"]] = (t, tuple(e["pos"]))
        self.recent_contacts = {i: entry for i, entry in self.recent_contacts.items()
                                if t - entry[0] <= 2.0}
        self.failed_pickups = {i: expiry for i, expiry in self.failed_pickups.items() if expiry > t}
        for event in obs.get("events", []):
            if event.get("type") == "kill":
                contact = self.recent_contacts.pop(event.get("target"), None)
                self.kill_site = contact[1] if contact else None
                self.landed_kills += 1
                self.kill_facing = me["aim"][0]
                self.aftermath_until = t + (4.0 if hp_ratio > 0.9 else 8.0)
            if event.get("type") == "effect":
                if event.get("effect") == "short":
                    self.short_until = t + 2.0
                if event.get("effect") == "concuss":
                    self.concuss_until = t + 0.5
            if event.get("type") == "took":
                self.swap_until = t + 2
            if event.get("type") == "hit":
                self.last_hit_id, self.last_hit_time = event.get("target"), t
            elif event.get("type") == "take_failed":
                self.failed_pickups[event.get("id")] = t + (5 if event.get("reason") == "gone" else 0.6)
            if event.get("type") == "damaged" and event.get("bearing") is not None:
                self.damage_burst = (self.damage_burst if t-self.last_damage_t<1.2 else 0) + event.get("amount",0)
                self.last_damage_t = t
                self.damage_bearing = event["bearing"]
                self.damage_tick = tick
                self.attacker = event.get("by")
                if event.get("rear"):
                    self.rear_time = t
        for s in obs.get("sounds", []):
            heard_at = t - s.get("age", 0)
            if s.get("kind") == "reload" and heard_at > self.reload_time:
                self.reload_time, self.reload_sound = heard_at, s["bearing"]
            if s.get("kind") in ("jet","boost") and heard_at > self.jet_time:
                self.jet_time, self.jet_sound = heard_at, s["bearing"]
            if s.get("kind") == "step" and heard_at > self.step_time:
                self.step_time, self.step_bearing = heard_at, s["bearing"]
            if (s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion")
                    and s.get("dist") in ("near", "mid") and s.get("age", 0) < 0.5):
                self.listen_bearing = s["bearing"]
                self.listen_tick = tick

        offer = obs.get("offer")
        if offer:
            bonuses = {}
            if enemy and abs(wrap(math.atan2(enemy["pos"][1]-y, enemy["pos"][0]-x)-me["aim"][0])) > 1:
                bonuses["gyros"] = 99
            if not me.get("in_zone", True):
                bonuses["servos"] = 106
            if weapons[0] and weapons[0].get("ammo", 12) <= 2:
                bonuses["gun_mag"] = 98
            if "lightning" in held and energy < 40:
                bonuses["lightning_efficiency"] = 102
            if "mortar" in held and len(enemies) >= 2 and distance(enemies[0]["pos"], enemies[1]["pos"]) < 16:
                bonuses["mortar_splash"] = 103
            pick = self.choose_upgrade(offer, hp_ratio, weapons[1] is not None, bonuses)
            if pick is not None:
                action["pick"] = pick

        votes = {kind: sum(distance(pos, e["pos"]) <= 150 and beaten in e.get("weapons", [])
                           for e in enemies) for kind, beaten in BEATS.items()}
        for seen, contact, _ in self.tracks.values():
            if seen < t and distance(pos,contact["pos"])<150:
                for kind,beaten in BEATS.items():
                    votes[kind] += .45*max(0,1-(t-seen)/3)*(beaten in contact.get("weapons",[]))
        if next_radius<CONFIG["terminal_circle"]:
            votes["lightning"]+=.8*(energy>40)
            votes["mortar"]+=.65
        if energy<22 and "lightning" in votes:
            votes["lightning"] -= 0.5
        if t < self.short_until:
            votes["laser"] -= 0.5
        second = weapons[1]
        # In the final ring retaining a contested upgrade denies that pickup.
        if self.alive<=4:
            for p in obs.get("pickups",[]):
                if p["kind"] in votes and any(distance(p["pos"],e["pos"])<20 for e in enemies):
                    votes[p["kind"]]+=.4
        pickups = [p for p in obs.get("pickups", []) if p["id"] not in self.gone and p["id"] not in self.failed_pickups and p["kind"] in votes and p["kind"] not in held
                   and abs(z - p["pos"][2]) < 3.5 and (not second or
                   (t >= self.swap_until and not second.get("charge") and votes[p["kind"]] > votes.get(second["kind"], 0)))]
        nearby = [p for p in pickups if distance(pos, p["pos"]) < self.pickup_radius]
        if nearby:
            ranks = {"lightning": 2, "laser": 1, "mortar": 0}
            if next_radius <= 45 and energy>=30:
                ranks["lightning"] += 2
            elif enemy_dist>100:
                ranks["laser"] += 2
            best = min(nearby, key=lambda p: (-votes[p["kind"]], -ranks[p["kind"]], distance(pos, p["pos"])))
            action["take"] = best["id"]
            if second:
                action["drop"] = 1

        # ---- zone: time-aware urgency instead of "always sprint to the next center"
        d_next = distance(pos, next_center)
        safe_r = max(0.0, next_radius - min(25.0, next_radius * 0.3))
        outside_now = (not me.get("in_zone", True)) or distance(pos, center) > radius - 12
        need_t = max(0.0, d_next - safe_r) / walk
        tts = zone.get("shrink_start", 1e9) - t
        phase = (tuple(next_center), next_radius)
        if phase != self.zone_key:
            self.zone_key = phase
            self.escape_until = -1
            self.path = []
            self.plan_tick = -999
        if self.nav and d_next > safe_r and not self.nav.walk_clear(pos, next_center):
            need_t *= 1.45
        start = max(t, zone.get("shrink_start", t))
        end = zone.get("shrink_end", start)
        f = min(1., max(0., t+3-start) / max(0.001, end-start))
        forecast = [a+(b-a)*f for a,b in zip(center,next_center)]
        boundary_soon = distance(pos, forecast) > radius + f*(next_radius-radius)
        urgent = boundary_soon or outside_now or (d_next > safe_r and (tts <= 0 or tts < need_t * 1.3 + 12))

        # Resting spot: a ring inside the next zone, away from the crowd at its center.
        hold_r = max(0.0, 0.55 * next_radius - 10.0) if (SETTLE and next_radius > 60) else 0.0
        if self.alive<=2:hold_r=0
        elif hold_r and self.nav:
            hold_r *= .8 if self.covered(pos)<2 else 1.08
        goal = None
        arrival_margin = min(10.0, radius * 0.5)
        if d_next > hold_r + arrival_margin:
            k = hold_r / d_next if d_next > 0.001 else 0.0
            goal = (next_center[0] + (x - next_center[0]) * k, next_center[1] + (y - next_center[1]) * k)

        if goal and not urgent and not enemy and self.nav and hold_r and tick%6<2:
            # Score several inward landing spots for cover and the short-side approach.
            a=math.atan2(y-next_center[1],x-next_center[0])
            candidates=[(next_center[0]+hold_r*math.cos(a+b),next_center[1]+hold_r*math.sin(a+b)) for b in (0,-.35,.35)]
            goal=max(candidates,key=lambda q:self.covered(q)-.035*distance(pos,q)+.012*distance(q,center))
        if goal and not urgent and enemy and hold_r > 10:
            # Shift the resting point away from the nearest visible contact.
            ax, ay = unit(goal[0]-enemy["pos"][0], goal[1]-enemy["pos"][1])
            alternative = (goal[0]+12*ax, goal[1]+12*ay)
            if distance(alternative,next_center) <= safe_r:
                goal = alternative
        pursuing = None
        if DROPS and not urgent and enemy_dist > 90:
            best_drop = None
            best_time = float("inf")
            for dr in obs.get("drops", []):
                dd = distance(pos, dr["pos"])
                if dd > 140 or distance(dr["pos"], next_center) > safe_r:
                    continue
                if any(distance(e["pos"], dr["pos"]) < 50 for e in enemies):
                    continue
                holder = dr.get("holder")
                if holder not in (None, my_id):
                    continue
                radius_to_hold = max(0.0, dr.get("radius", 8) - 2)
                inside = dd <= radius_to_hold
                # A landed drop restores full health. Finish a nearby capture
                # while wounded, but keep the usual health gate for travel.
                lands_in = dr.get("lands_in", 0)
                midpoint = ((x+dr["pos"][0])/2, (y+dr["pos"][1])/2)
                if not inside and any(distance(e["pos"],midpoint)<45 for e in enemies):
                    continue
                if not inside and tick-self.damage_tick < 30:
                    continue
                if hp_ratio <= 0.5 and not (dd <= radius_to_hold+12 and lands_in <= 0):
                    continue
                if not inside and self.nav and enemy_dist<135 and self.covered(dr["pos"])==0:
                    continue
                if lands_in > dd / walk + 6:
                    continue
                travel = max(0.0, dd - radius_to_hold) / walk
                progress = dr.get("progress", 0) if holder == my_id else 0.0
                capture_left = max(0.0, dr.get("capture_s", 4) - progress)
                finish = max(travel, lands_in) + capture_left
                if distance(dr["pos"],center)>radius-max(1,dr.get("radius",8)/2):
                    continue
                if d_next > safe_r and finish + need_t + 3 > tts:
                    continue
                if dr.get("id") == self.wanted_drop:
                    finish -= 0.6
                if hp_ratio<0.4 and enemy_dist<120 and not inside:
                    continue
                if finish < best_time:
                    best_time = finish
                    best_drop = dr
            if best_drop is not None:
                self.wanted_drop = best_drop["id"]
                pursuing = "drop"
                drop_pos = best_drop["pos"]
                inside = distance(pos, drop_pos) <= max(0.0, best_drop.get("radius", 8) - 2)
                goal = None if inside else tuple(drop_pos[:2])
                slack=best_drop.get("lands_in",0)-distance(pos,drop_pos)/walk
                if slack>3 and not inside:
                    stops=[u["pos"] for u in self.cubes.near(pos,18) if
                           (distance(pos,u["pos"])+distance(u["pos"],drop_pos))/walk<best_drop["lands_in"]-1 and
                           (not self.nav or self.nav.walk_clear(pos,u["pos"]))]
                    if stops:goal=min(stops,key=lambda q:distance(pos,q))
        # Only interrupt normal looting briefly, and only for a confirmed kill.
        aftermath = self.alive>2 and t < self.aftermath_until
        safe_pause = (aftermath and not urgent and enemy_dist > 100
                      and tick - self.damage_tick > 20 and not obs.get("projectiles")
                      and t - self.step_time > 1 and tick - self.listen_tick > 15)
        salvage_window = (aftermath and not urgent and enemy_dist>100 and tick-self.damage_tick>20
                          and t-self.step_time>1 and tick-self.listen_tick>15
                          and not any(distance(pos,q["pos"])<40 for q in obs.get("projectiles",[])))
        if salvage_window and pursuing is None and self.kill_site and self.nav is not None:
            cubes = [cube["pos"] for cube in obs.get("cubes", [])
                     if distance(cube["pos"], self.kill_site) <= 12
                     and distance(pos, cube["pos"]) <= 32
                     and abs(z - cube["pos"][2]) < 3.5
                     and distance(cube["pos"], next_center) < safe_r
                     and not any(distance(e["pos"],cube["pos"]) < 80 for e in enemies)
                     and not any(distance(cube["pos"],q["pos"]) < 15 for q in obs.get("projectiles",[]))
                     and self.nav.walk_clear(pos, cube["pos"])]
            if cubes:
                # Prefer a pickup that brings several cubes within collection range.
                def haul_value(point):
                    count = sum(distance(point, other) <= self.cube_radius for other in cubes)
                    route_cost = distance(pos,point) / count
                    if hp_ratio < 0.5:
                        route_cost += 0.4*distance(pos,point)
                    if distance(point,next_center)<d_next:
                        route_cost -= 1
                    return route_cost
                def tour(start):
                    remaining=[q for q in cubes[:12] if q!=start];q=start
                    cost=haul_value(start)
                    for _ in range(min(3,len(remaining))):
                        nxt=min(remaining,key=lambda v:distance(v,q))
                        cost+=.45*distance(q,nxt);remaining.remove(nxt);q=nxt
                    return cost
                goal = min(cubes[:12],key=tour)[:2]
                pursuing = "salvage"
        threshold_rush=me.get("cubes",0)+1 in self.upgrade_thresholds and enemy_dist>90 and hp_ratio>.6
        if pursuing is None and (not aftermath or threshold_rush) and not urgent and hp_ratio > 0.55 and enemy_dist > 75:
            cands = []
            for cube in self.cubes.near(pos,90):
                d = distance(pos, cube["pos"])
                if d < 90 and abs(z - cube["pos"][2]) < 3.5 and distance(cube["pos"], next_center) < safe_r:
                    if d / walk + need_t + 3 > tts and d_next > safe_r:
                        continue
                    if me.get("cubes",0) >= 10 and d > 24:
                        continue
                    cluster = sum(1 for _ in self.cubes.near(cube["pos"],8))
                    danger = any(distance(e["pos"],cube["pos"]) < 75 for e in enemies)
                    priority_distance = d / (1 + 0.25*min(3,max(0,cluster-1))) + 35*danger
                    if me.get("cubes", 0) + 1 in self.upgrade_thresholds:
                        priority_distance *= 0.55
                    cands.append((priority_distance, cube["pos"]))
            if pickups:
                for item in pickups:
                    d = distance(pos, item["pos"])
                    if d < 90 and distance(item["pos"], next_center) < safe_r:
                        if self.nav and self.nav.cell_of(*item["pos"][:2]) is None:
                            continue
                        if any(distance(e["pos"],item["pos"])+5 < d for e in enemies):
                            continue
                        unarmed_bonus = 14 if not second and d<35 else 0
                        item_pos=item["pos"]
                        if item["id"] in self.retry:
                            nx,ny=unit(item_pos[0]-x,item_pos[1]-y)
                            item_pos=(item_pos[0]+nx,item_pos[1]+ny,item_pos[2])
                        cands.append((d+8-min(15,5*votes[item["kind"]])-unarmed_bonus,item_pos))
            if cands:
                goal = min(cands, key=lambda c: c[0])[1][:2]
                pursuing = "loot"
        farm = None
        if FARM and self.alive>3 and t-self.last_seen>CONFIG["farm_quiet"] and not any(t-st<CONFIG["farm_quiet"] for st,*_ in self.ears) and not aftermath and tick-self.listen_tick > 30 and t-self.step_time > 1.0 and pursuing is None and not urgent and hp_ratio > 0.7 and enemy_dist > 100:
            best = None
            for n in npcs:
                if sum(distance(n["pos"], other["pos"]) < 30 for other in npcs) > 1:
                    continue
                d = distance(pos, n["pos"])
                if d < 55 and abs(z - n["pos"][2]) < 6 and (best is None or d < best[0]):
                    best = (d, n, 1.75)
            for c in obs.get("crates", []):
                d = distance(pos, c["pos"])
                if d >= 40 or (self.nav and not self.nav.shot_clear((x,y,muzzle_z),(*c["pos"][:2],c["pos"][2]+1.5))):
                    continue
                value = d - (8 if c.get("hp",300) < 150 else 0)
                if c["id"] == self.last_hit_id and t-self.last_hit_time < 1:
                    value -= 10
                if (d < 40 and abs(z - c["pos"][2]) < 6
                        and (best is None or value < best[0] or
                             (not weapons[1] and best[1].get("kind") == "npc" and d < best[0] + 12))):
                    best = (value, c, 1.5)
            if best and distance(best[1]["pos"], next_center) < safe_r:
                farm = best
                pursuing = "farm"
                goal = (best[1]["pos"][0], best[1]["pos"][1]) if best[0] > 32 else None

        if hp_ratio < 0.50:
            self.recovering = True
        elif hp_ratio > 0.78:
            self.recovering = False
        posting = False
        if enemy and enemy_dist > 35 and not urgent and not pursuing and self.nav and abs(z-self.nav.ground(x,y)) < 2:
            key = (phase,enemy["id"],self.recovering, round(enemy["pos"][0]/12), round(enemy["pos"][1]/12))
            if key != self.post_key or tick-self.post_tick >= 18:
                self.post = self.select_post(pos,enemy,next_center,safe_r,self.recovering or t<max(self.short_until,self.concuss_until) or bool(weapons[0] and weapons[0].get("reload",0)>.2),enemies,obs.get("projectiles", []))
                self.post_tick,self.post_key = tick,key
            if self.post and not self.nav.walk_clear(pos,self.post) and distance(pos,self.post)>2:
                self.post = None
                self.post_tick = tick-16
            if self.post and urgent is False and distance(self.post,center)>radius-1:
                self.post = None
            if self.post:
                posting = True
                goal = self.post if distance(pos,self.post)>2 else None
        else:
            self.post_key = None
        if second and not nearby and not urgent and not enemies and not second.get("charge") and t>=self.swap_until:
            upgrade=next((u for u in pickups if distance(pos,u["pos"])<=self.pickup_radius+1.5 and votes[u["kind"]]>votes.get(second["kind"],0)+1 and (not self.nav or self.nav.walk_clear(pos,u["pos"]))),None)
            if upgrade:
                goal=upgrade["pos"][:2];pursuing="loot"
                action["drop"]=1
        goal=self.macro(goal,pursuing,urgent,enemy,self.recovering or t<max(self.short_until,self.concuss_until) or self.dps>CONFIG["sustained_fraction"]*me["max_hp"],posting)
        gx, gy = (0.0, 0.0) if goal is None else self.steer(pos, goal, tick)

        # ---- fight / flee / strafe
        engaged = enemy is not None and enemy_dist < 170
        fighting_move = False
        mx, my = gx, gy
        if engaged and not urgent and not posting:
            ex, ey = enemy["pos"][0], enemy["pos"][1]
            ax, ay = unit(x - ex, y - ey)
            sx, sy = -ay, ax
            if tick >= self.strafe_until:
                self.strafe_sign = -self.strafe_sign
                self.strafe_until = tick + (self.rng.randint(5,10) if tick-self.damage_tick < 40 else self.rng.randint(14,24))
            sign = self.strafe_sign
            if len(enemies)>1 and tick >= self.strafe_until-4:
                left=(x+sx*12,y+sy*12)
                right=(x-sx*12,y-sy*12)
                sign=1 if distance(left,enemies[1]["pos"])>distance(right,enemies[1]["pos"]) else -1
            if self.nav is not None and not self.nav.walk_clear((x, y), (x + sx * sign * 10, y + sy * sign * 10)):
                sign = -sign
                self.strafe_sign = sign
            flee = (t-self.last_damage_t<1 and self.damage_burst>CONFIG["burst_damage"]*me["max_hp"]) or self.should_flee(me, enemy, enemies, hp_ratio, held)
            if flee and enemy_dist < 110:
                mx, my = unit(0.75 * ax + 0.45 * sx * sign + 0.25 * gx,
                              0.75 * ay + 0.45 * sy * sign + 0.25 * gy)
                if distance((x + mx * 18, y + my * 18), next_center) > max(5, next_radius - 10):
                    mx, my = unit(next_center[0] - x, next_center[1] - y)
                fighting_move = True
            elif STRAFE and enemy_dist < 150:
                # Choose spacing from both loadouts. Keep the gun useful while
                # denying an enemy arc; close only for our own usable arc.
                their_weapons = enemy.get("weapons", [])
                own_arc = weapons[1] and weapons[1]["kind"] == "lightning" and energy > 25
                if own_arc and hp_ratio > 0.65 and len(enemies) == 1 and "mortar" not in their_weapons:
                    arc_range = weapons[1]["stats"].get("range", 45)
                    low, high = 0.55 * arc_range, 0.85 * arc_range
                elif "lightning" in their_weapons:
                    low, high = 65, 125
                else:
                    low, high = 45, 115
                away_w = 0.85 if enemy_dist < low else (-0.65 if enemy_dist > high else 0.0)
                mx, my = unit(0.9 * sx * sign + away_w * ax + 0.15 * gx,
                              0.9 * sy * sign + away_w * ay + 0.15 * gy)
                if distance((x + mx * 12, y + my * 12), next_center) > max(5, next_radius - 10):
                    cx_, cy_ = unit(next_center[0] - x, next_center[1] - y)
                    mx, my = unit(mx + cx_, my + cy_)
                fighting_move = True
        # Check the combined strafe/retreat direction after adding radial motion.
        if fighting_move and self.nav and abs(z - self.nav.ground(x, y)) < 2:
            end = (x + 8 * mx, y + 8 * my)
            if not self.nav.walk_clear((x, y), end):
                heading = math.atan2(my, mx)
                mx = my = 0.0
                for offset in (0.55, -0.55, 1.1, -1.1, 1.65, -1.65, math.pi):
                    vx, vy = math.cos(heading + offset), math.sin(heading + offset)
                    end = (x + 8 * vx, y + 8 * vy)
                    if (distance(end, center) <= radius and
                            distance(end, next_center) <= max(d_next, safe_r) and
                            self.nav.walk_clear((x, y), end)):
                        mx, my = vx, vy
                        break
        gx, gy = mx, my

        dodge = None
        if obs.get("projectiles"):
            if not urgent:
                dodge = self.dodge(me, obs["projectiles"], next_center, safe_r)
            elif self.nav is not None and (gx or gy):
                current_distance = distance(pos, center)

                def rotation_options(sides):
                    viable = []
                    for sx, sy in sides:
                        # Keep a lateral component even for head-on projectiles.
                        vx, vy = unit(sx + 0.65 * gx, sy + 0.65 * gy)
                        end = (x + 10 * vx, y + 10 * vy)
                        if self.recovering and any(distance(end,e["pos"])<20 and distance(end,e["pos"])<distance(pos,e["pos"]) for e in enemies):
                            continue
                        if distance(end, next_center) >= d_next:
                            continue
                        end_distance = distance(end, center)
                        if outside_now and end_distance >= current_distance:
                            continue
                        if not outside_now and end_distance > radius:
                            continue
                        if self.nav.walk_clear((x, y), end):
                            viable.append((vx, vy))
                    return viable

                dodge = self.dodge(me, obs["projectiles"], next_center, safe_r, rotation_options)
            if dodge:
                gx, gy = dodge

        # ---- stuck detection: really not moving while we ask to move
        if urgent and dodge:
            self.escape_until = tick
            self.moving_obs = 0
            self.hist.clear()
        elif math.hypot(gx, gy) > 0.3:
            self.moving_obs += 1
        else:
            self.moving_obs = 0
            self.hist.clear()
        self.hist.append((x, y))
        if self.moving_obs >= 10 and len(self.hist) == 10 and tick >= self.escape_until:
            h = list(self.hist)
            net = distance(h[0], h[-1])
            plen = sum(distance(h[i], h[i + 1]) for i in range(9))
            if net < 2.0 and (plen < 4.0 or not fighting_move):
                d = unit(gx, gy)
                if self.nav is not None:
                    self.nav.learn_block(x, y, d, tick)
                self.path = []
                self.plan_tick = -999
                ang = math.atan2(d[1], d[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(1.6, 2.4)
                self.escape_dir = (math.cos(ang), math.sin(ang))
                self.escape_until = tick + 14
                self.hist.clear()
                self.moving_obs = 0
        escaping = tick < self.escape_until
        if escaping:
            gx, gy = self.escape_dir

        if (not urgent and not escaping and not dodge and enemy is None and pursuing is None
                and (t-self.step_time < 1.5 or tick-self.listen_tick < 20)):
            pace = min(1, 0.85 * self.step_speed / walk)
            gx, gy = gx * pace, gy * pace
        action["move"] = [gx, gy]
        action["boost"] = bool((energy > 45 and (dodge or outside_now or (urgent and d_next > 90)))
                               or (escaping and energy > 35))
        if fighting_move and energy > 82 and enemy_dist < 85 and me.get("grounded",True) and (gx or gy):
            action["boost"] = True
        if pursuing == "drop" and goal and distance(pos,goal)>25 and energy>85 and enemy_dist>120:
            action["boost"] = True
        if "lightning" in held and energy < 52 and not urgent and not dodge and not escaping:
            action["boost"] = False
        ledge = False
        if self.nav and (gx or gy) and not escaping and energy>55 and t>=self.concuss_until:
            cell = self.nav.cell_of(x+4*gx,y+4*gy)
            ledge = cell is not None and z+1.2 < self.nav.top[cell] < z+4
        action["jump"] = bool(me.get("grounded", True) and (escaping or ledge))

        tgt=None
        clear=False
        try:
            # ---- aim and fire
            default_ranges = {"gun": 230, "laser": 280, "lightning": 45}
            reach = max((w["stats"].get("range", default_ranges[w["kind"]])
                         for w in weapons if w and w["kind"] in default_ranges), default=0)
            gun, second = weapons[0], weapons[1]
            tgt = None
            # Keep movement keyed to the closest mech. Aim at the closest one
            # with a clear firing lane; remembered targets can be behind walls.
            best_cost = float("inf")
            for contact in enemies:
                if tgt is not None and time.perf_counter()>=self.deadline:break
                if distance(pos, contact["pos"]) >= reach:
                    break
                height = 4.5 * (1 + 0.05 * contact.get("cubes", 0))
                target_pos = contact["pos"]
                if self.nav and not self.nav.shot_clear(
                    (x, y, muzzle_z), (*target_pos[:2], target_pos[2] + height)
                ):
                    continue
                d = distance(pos, target_pos)
                hp = contact.get("hp")
                damage = max(1,gun["stats"].get("damage",45)) if gun else 45
                cost = (hp if hp is not None else contact.get("max_hp") or 1400) / damage + d / 35
                if "laser" in held and "mortar" in contact.get("weapons",[]):
                    cost -= 3
                if "lightning" in held and d<40 and "laser" in contact.get("weapons",[]):
                    cost -= 4
                if "mortar" in held and 35<d<130 and "lightning" in contact.get("weapons",[]):
                    cost -= 2
                if d<22 and self.recovering:
                    cost -= 15
                bearing = math.atan2(target_pos[1]-y,target_pos[0]-x)
                cost += 8*abs(wrap(bearing-me["aim"][0]))/max(.1,stats.get("turn_rate",4))
                if hp is not None and hp<.5*(contact.get("max_hp") or 1400) and contact["id"] in self.collateral:cost-=10
                if "aim" in contact and abs(wrap(contact["aim"][0]-bearing)) < math.pi/2:
                    cost -= 3
                if hp is not None and gun and hp <= gun["stats"].get("damage",45):
                    cost -= 8
                if contact["id"] == self.last_hit_id and t-self.last_hit_time < 1.5:
                    cost -= 3
                if "mortar" in held:
                    cost -= 2 * min(3,sum(e["id"] != contact["id"] and distance(e["pos"],target_pos)<7 for e in enemies))
                if contact["id"] == self.focus_id:
                    cost -= 3
                if contact["id"] == self.attacker and tick-self.damage_tick < 30:
                    cost -= 8
                if contact.get("charging") and "aim" in contact:
                    to_us = math.atan2(y-target_pos[1], x-target_pos[0])
                    if abs(wrap(to_us-contact["aim"][0])) < 0.20:
                        cost -= 10
                if cost < best_cost:
                    best_cost, tgt = cost, (contact, height, True)
            self.focus_id = tgt[0]["id"] if tgt else None
            if tgt is None and enemy and enemy_dist < reach:
                tgt = (enemy, 4.5 * (1 + 0.05 * enemy.get("cubes", 0)), True)
            elif tgt is None and farm and farm[0] < 90:
                tgt = (farm[1], farm[2], False)

            if tgt:
                obj, h, is_mech = tgt
                tp = obj["pos"]
                tv = obj.get("vel", (0.0, 0.0, 0.0))
                dist = distance(pos, tp)
                speed = gun["stats"].get("speed", 252) if gun and gun["kind"] == "gun" else 252
                instant = bool(is_mech and second and second.get("cooldown", 0) <= 0 and (
                    (second["kind"] == "laser" and t >= self.short_until and dist < second["stats"].get("range", 280))
                    or (second["kind"] == "lightning" and energy > 20
                        and dist < second["stats"].get("range", 45) - 2)))
                travel = 0.0
                if not instant:
                    for _ in range(2):
                        travel = min(gun["stats"].get("range",230)/max(speed,1) if gun else 0.9, math.hypot(tp[0] + tv[0] * travel - x,
                                                     tp[1] + tv[1] * travel - y) / max(speed, 1.0))
                lx, ly = tp[0] + tv[0] * travel, tp[1] + tv[1] * travel
                lz = tp[2] + h + tv[2] * travel
                track=self.tracks.get(obj.get("id"))
                if track and travel:
                    horizon=min(travel,.45)
                    lx+=.5*track[2][0]*horizon**2
                    ly+=.5*track[2][1]*horizon**2
                    if abs(tv[2])>2:lz-=.5*self.game.get("mech",{}).get("gravity",22)*travel**2
                if self.nav:
                    lz = max(lz,self.nav.ground(lx,ly)+h)
                yaw = math.atan2(ly - y, lx - x)
                pitch = math.atan2(lz - muzzle_z, max(0.01, math.hypot(lx - x, ly - y)))
                action["aim"] = [yaw, pitch]
                yaw_err = wrap(yaw - me["aim"][0])
                pitch_err = pitch - me["aim"][1]
                aligned = (abs(yaw_err) < max(0.03, math.atan2(2.3, dist))
                           and abs(pitch_err) < max(0.04, math.atan2(3.5, dist)))
                if dist>120 and gun and gun.get("ammo",12)<5:
                    aligned=aligned and abs(yaw_err)<max(.012,math.atan2(2,dist))
                clear = True
                if self.nav is not None and dist > 15:
                    clear = self.nav.shot_clear((x, y, muzzle_z), (tp[0], tp[1], tp[2] + h))
                healing = (self.recovering and me.get("in_zone", True) and enemy_dist > 85
                           and tick-self.damage_tick > 30 and not outside_now and
                           not (is_mech and obj.get("charging")))
                recovering_after_kill = (safe_pause and hp_ratio < 0.78 and is_mech
                                         and not obj.get("charging") and not action["boost"])
                healing = healing or recovering_after_kill
                if gun and gun["kind"] == "gun":
                    gun_range = gun["stats"].get("range", 230) if is_mech else 90
                    gun_clear = clear
                    if gun_clear and self.nav and travel>0.05:
                        gun_clear = self.nav.shot_clear((x,y,muzzle_z),(lx,ly,lz))
                    action["fire"][0] = bool(not healing and aligned and gun_clear and math.hypot(dist,lz-muzzle_z) < gun_range and gun.get("ammo", 1) > 0)
                    if gun.get("ammo",12) <= 2 and dist > 80 and abs(yaw_err) > 0.025:
                        action["fire"][0] = False
                    can_top_up = (is_mech and not clear and enemy_dist > 60 and not urgent
                                  and tick - self.damage_tick > 40 and tick - self.listen_tick > 40
                                  and gun.get("ammo", 1) <= gun["stats"].get("mag", 12) // 2)
                    if safe_pause and not action["fire"][0] and not clear:
                        can_top_up = gun.get("ammo", 1) < gun["stats"].get("mag", 12)
                    if (gun.get("ammo", 1) == 0 or can_top_up) and gun.get("reload", 0) <= 0:
                        action["reload"] = True
                if (not healing and is_mech and second and second.get("cooldown", 0) <= 0 and (clear or second["kind"]=="mortar")
                        and (second["kind"] != "laser" or t >= self.short_until)):
                    kind, st = second["kind"], second["stats"]
                    if kind == "lightning":
                        near_aim = abs(yaw_err) < math.radians(st.get("cone_deg",30))
                        for obstacle in obs.get("crates", []) + npcs:
                            ox,oy = obstacle["pos"][:2]
                            if distance(pos,obstacle["pos"]) < dist and abs(wrap(math.atan2(oy-y,ox-x)-me["aim"][0])) < 0.2:
                                range_chain=st.get("chain_range",14)
                                near_aim=bool(distance(obstacle["pos"],tp)<range_chain and (not self.nav or self.nav.shot_clear((*obstacle["pos"][:2],obstacle["pos"][2]+2),(*tp[:2],tp[2]+h))))
                                break
                        action["fire"][1] = bool(near_aim and dist < st.get("range", 45) - 2 and energy > (8 if obj.get("charging") else 20))
                    elif kind == "mortar" and 25 < dist < 210:
                        splash=st.get("splash_radius",7)
                        cluster=next((e for e in enemies if e["id"]!=obj["id"] and distance(e["pos"],tp)<splash),None)
                        point=(tp[0],tp[1],tp[2]+(.8 if not clear else h))
                        if cluster:
                            point=tuple((p+q)/2 for p,q in zip(point,(*cluster["pos"][:2],cluster["pos"][2]+.8)))
                        arc = self.lob(x, y, muzzle_z + 2.5, point,
                                       tv, st.get("speed", 72), self.mortar_g)
                        if arc:
                            impact_delay = arc[2]
                            predicted_self = (x+me["vel"][0]*impact_delay,y+me["vel"][1]*impact_delay)
                            predicted_target = (tp[0]+tv[0]*impact_delay,tp[1]+tv[1]*impact_delay)
                            if distance(predicted_self,predicted_target) < st.get("splash_radius",7)+5:
                                arc = None
                        if arc:
                            yaw, pitch, _ = arc
                            action["aim"] = [yaw, pitch]
                            action["fire"][0] = False
                            action["fire"][1] = abs(wrap(yaw-me["aim"][0])) < 0.08 and abs(pitch-me["aim"][1]) < 0.06
                    elif kind == "laser" and dist < st.get("range", 280):
                        fraction = 0.72
                        target_hp = obj.get("hp")
                        if target_hp and "mortar" in obj.get("weapons",[]):
                            target_hp /= 1.6
                        if target_hp and target_hp <= st.get("damage", 315) * st.get("min_frac", 0.35):
                            fraction = 0.0
                        release_at = max(st.get("min_charge", 0.15), fraction * st.get("charge", 1))
                        # hold to charge; let go only when charged enough AND lined up
                        action["fire"][1] = not (second.get("charge", 0) >= release_at and aligned)
            else:
                if self.damage_bearing is not None and tick - self.damage_tick < 30:
                    yaw = self.damage_bearing
                elif self.listen_bearing is not None and tick - self.listen_tick < 15:
                    yaw = self.listen_bearing
                elif aftermath and t-self.jet_time < 0.5:
                    yaw = self.jet_sound
                elif t-self.reload_time < 1.2:
                    yaw = self.reload_sound
                elif t-self.jet_time < 0.9:
                    yaw = self.jet_sound
                elif self.step_bearing is not None and t-self.step_time < 0.8:
                    yaw = self.step_bearing
                elif aftermath and not (gx or gy):
                    yaw = self.kill_facing + math.pi + 0.65*math.sin(t*3)
                elif gx or gy:
                    yaw = self.scan(pos,next_center,t)
                else:
                    yaw = self.scan(pos,next_center,t)
                action["aim"] = [yaw, 0.0]
                if gun and gun["kind"] == "gun" and (t-self.step_time > 1 or gun.get("ammo",0)==0) and gun.get("ammo", 1) < gun["stats"].get("mag", 12):
                    action["reload"] = True
            if aftermath and hp_ratio < 0.45 and enemy is None and not urgent and not dodge:
                action["boost"] = False
            if (aftermath and self.landed_kills >= 2 and action.get("reload") and gun and
                    gun.get("ammo",0)>3 and t-self.jet_time<1.2):
                action.pop("reload",None)
            if t-self.rear_time < 0.65 and self.damage_bearing is not None and not any(action["fire"]):
                action["aim"] = [self.damage_bearing, 0.0]
        except (ValueError,TypeError,ArithmeticError,KeyError,IndexError) as exc:
            self.log({"aim":str(exc)})
            action["fire"]=[False, False]
        return self.finish(obs,action,urgent,enemy,dodge,farm,goal,need_t,tts,escaping,clear if tgt else False,tgt)


class Bot(Decisions, CenturyBot):
    name = "Darwin Century Atlas"

if __name__ == "__main__":
    mf.run(Bot)
