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

# Feature flags. Flip ONE at a time between submissions to see what helps.
USE_NAV = True    # A* over the campus map instead of walking straight lines
SETTLE = True     # wait on a ring inside the next zone, not at its exact center
STRAFE = True     # keep moving sideways while fighting
FARM = True       # shoot nearby crates / NPC drones for cubes when it is quiet
DROPS = True      # go for supply drops when it looks safe
PLAN_BUDGET_S = 0.018   # max wall time for one A* search (tick budget is 50 ms)
MAP_BUDGET_S = 8.0      # max time spent building the nav grid at startup

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


class NavGrid:
    """Coarse walkability grid built from v2/world.json prisms and terrain."""
    CELL = 3.0

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
            if idx is not None and len(self.learned) < 3000:
                self.learned[idx] = tick + 600

    def walk_clear(self, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(1, int(math.hypot(dx, dy) / 1.5))
        for k in range(1, n + 1):
            f = k / n
            idx = self.cell_of(a[0] + dx * f, a[1] + dy * f)
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
            if idx is not None and self.top[idx] > a[2] + dz * f + 0.5:
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
            if (n & 127) == 0 and time.perf_counter() - t0 > budget_s:
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
                if dx and dy and (wblock[cy * gnx + xx] == 1 or wblock[yy * gnx + cx] == 1):
                    continue
                ng = gi + c * (4.0 if w == 2 else 1.0)
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
            while k + 1 < len(pts) and self.walk_clear(pts[i], pts[k + 1]):
                k += 1
            out.append(pts[k])
            i = k
        return out


class Bot:
    name = "Darwin Second Wind"

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
        self.damage_bearing = None
        self.damage_tick = -30
        self.listen_bearing = None
        self.listen_tick = -30
        self.strafe_sign = 1
        self.strafe_until = 0

    # ------------------------------------------------------------------ helpers
    def choose_upgrade(self, offer, hp_ratio=1.0, has_secondary=False):
        choices = offer.get("choices", [])
        terms = [("armor", 100), ("repair", 95), ("regen", 90), ("health", 85), ("hp", 85),
                 ("speed", 70), ("energy", 65), ("shield", 60), ("sensor", 55)]

        preferred = {"armor": 101, "plating": 98, "nanites": 104 if hp_ratio < 0.65 else 89,
                     "gun_damage": 95 if not has_secondary else 81, "gun_rate": 80,
                     "gun_mag": 79, "laser_damage": 84, "lightning_range": 84}

        def priority(c):
            if c.get("id") in preferred:
                return preferred[c["id"]]
            s = (c.get("id", "") + " " + c.get("name", "") + " " + c.get("desc", "")).lower()
            return max([sc for word, sc in terms if word in s] or [10])
        return max(range(len(choices)), key=lambda i: priority(choices[i])) if choices else None

    def dodge(self, me, projectiles, center, safe_radius):
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
            if math.hypot(dx, dy) >= margin or abs(pz - z - 4.5 + (vz - mz) * t) > margin:
                continue
            sx, sy = unit(-vy, vx)
            options = [(sx, sy), (-sx, -sy)]
            safe = [v for v in options if distance((x + 10 * v[0], y + 10 * v[1]), center) <= safe_radius]
            if safe:
                direction = max(safe, key=lambda v: (
                    -(dx * v[0] + dy * v[1]), -distance((x + 10 * v[0], y + 10 * v[1]), center)))
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
            if distance((x, y), goal) < 10 and nav.walk_clear((x, y), goal):
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
            if distance((x, y), p[j]) < 1.0:
                return direct
            return unit(p[j][0] - x, p[j][1] - y)
        except Exception:
            return direct

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

        for event in obs.get("events", []):
            if event.get("type") == "damaged" and event.get("bearing") is not None:
                self.damage_bearing = event["bearing"]
                self.damage_tick = tick
        for s in obs.get("sounds", []):
            if (s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion")
                    and s.get("dist") in ("near", "mid") and s.get("age", 0) < 0.5):
                self.listen_bearing = s["bearing"]
                self.listen_tick = tick

        offer = obs.get("offer")
        if offer:
            pick = self.choose_upgrade(offer, hp_ratio, weapons[1] is not None)
            if pick is not None:
                action["pick"] = pick

        # Grab a second weapon when in range; prefer lightning or laser.
        pickups = [p for p in obs.get("pickups", []) if p["kind"] not in held
                   and p["kind"] != "mortar" and abs(z - p["pos"][2]) < 3.5]
        nearby = [p for p in pickups if distance(pos, p["pos"]) < self.pickup_radius]
        if nearby and weapons[1] is None:
            nearby.sort(key=lambda p: (-(3 if p["kind"] == "lightning" else 2 if p["kind"] == "laser" else 1),
                                       distance(pos, p["pos"])))
            action["take"] = nearby[0]["id"]

        # ---- zone: time-aware urgency instead of "always sprint to the next center"
        d_next = distance(pos, next_center)
        safe_r = max(8.0, next_radius - 25.0)
        outside_now = (not me.get("in_zone", True)) or distance(pos, center) > radius - 12
        need_t = max(0.0, d_next - safe_r) / walk
        tts = zone.get("shrink_start", 1e9) - t
        urgent = outside_now or (d_next > safe_r and (tts <= 0 or tts < need_t * 1.3 + 12))

        # Resting spot: a ring inside the next zone, away from the crowd at its center.
        hold_r = max(0.0, 0.55 * next_radius - 10.0) if (SETTLE and next_radius > 60) else 0.0
        goal = None
        arrival_margin = min(10.0, radius * 0.5)
        if d_next > hold_r + arrival_margin:
            k = hold_r / d_next if d_next > 0.001 else 0.0
            goal = (next_center[0] + (x - next_center[0]) * k, next_center[1] + (y - next_center[1]) * k)

        pursuing = None
        if DROPS and not urgent and hp_ratio > 0.5 and enemy_dist > 90:
            for dr in obs.get("drops", []):
                dd = distance(pos, dr["pos"])
                if dd > 140 or distance(dr["pos"], next_center) > safe_r:
                    continue
                if dr.get("holder") not in (None, my_id):
                    continue
                if dr.get("lands_in", 0) > dd / walk + 6:
                    continue
                pursuing = "drop"
                goal = None if dd <= dr.get("radius", 8) - 2 else (dr["pos"][0], dr["pos"][1])
                break
        if pursuing is None and not urgent and hp_ratio > 0.55 and enemy_dist > 75:
            cands = []
            for cube in obs.get("cubes", []):
                d = distance(pos, cube["pos"])
                if d < 90 and abs(z - cube["pos"][2]) < 3.5 and distance(cube["pos"], next_center) < safe_r:
                    cands.append((d, cube["pos"]))
            if weapons[1] is None:
                for item in pickups:
                    d = distance(pos, item["pos"])
                    if d < 90 and distance(item["pos"], next_center) < safe_r:
                        cands.append((d + 8, item["pos"]))
            if cands:
                goal = min(cands, key=lambda c: c[0])[1][:2]
                pursuing = "loot"
        farm = None
        if FARM and pursuing is None and not urgent and hp_ratio > 0.7 and enemy_dist > 100:
            best = None
            for n in npcs:
                d = distance(pos, n["pos"])
                if d < 55 and abs(z - n["pos"][2]) < 6 and (best is None or d < best[0]):
                    best = (d, n, 1.75)
            for c in obs.get("crates", []):
                d = distance(pos, c["pos"])
                if d < 40 and abs(z - c["pos"][2]) < 6 and (best is None or d < best[0]):
                    best = (d, c, 1.5)
            if best and distance(best[1]["pos"], next_center) < safe_r:
                farm = best
                pursuing = "farm"
                goal = (best[1]["pos"][0], best[1]["pos"][1]) if best[0] > 32 else None

        gx, gy = (0.0, 0.0) if goal is None else self.steer(pos, goal, tick)

        # ---- fight / flee / strafe
        engaged = enemy is not None and enemy_dist < 170
        fighting_move = False
        mx, my = gx, gy
        if engaged and not urgent:
            ex, ey = enemy["pos"][0], enemy["pos"][1]
            ax, ay = unit(x - ex, y - ey)
            sx, sy = -ay, ax
            if tick >= self.strafe_until:
                self.strafe_sign = -self.strafe_sign
                self.strafe_until = tick + self.rng.randint(6, 14)
            sign = self.strafe_sign
            if self.nav is not None and not self.nav.walk_clear((x, y), (x + sx * sign * 10, y + sy * sign * 10)):
                sign = -sign
                self.strafe_sign = sign
            flee = self.should_flee(me, enemy, enemies, hp_ratio, held)
            if flee and enemy_dist < 110:
                mx, my = unit(0.75 * ax + 0.45 * sx * sign + 0.25 * gx,
                              0.75 * ay + 0.45 * sy * sign + 0.25 * gy)
                if distance((x + mx * 18, y + my * 18), next_center) > max(5, next_radius - 10):
                    mx, my = unit(next_center[0] - x, next_center[1] - y)
                fighting_move = True
            elif STRAFE and enemy_dist < 150:
                away_w = 0.6 if enemy_dist < 30 else (-0.3 if enemy_dist > 100 else 0.0)
                mx, my = unit(0.9 * sx * sign + away_w * ax + 0.15 * gx,
                              0.9 * sy * sign + away_w * ay + 0.15 * gy)
                if distance((x + mx * 12, y + my * 12), next_center) > max(5, next_radius - 10):
                    cx_, cy_ = unit(next_center[0] - x, next_center[1] - y)
                    mx, my = unit(mx + cx_, my + cy_)
                fighting_move = True
        gx, gy = mx, my

        dodge = None
        if not urgent and obs.get("projectiles"):
            dodge = self.dodge(me, obs["projectiles"], next_center, safe_r)
            if dodge:
                gx, gy = dodge

        # ---- stuck detection: really not moving while we ask to move
        if math.hypot(gx, gy) > 0.3:
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

        action["move"] = [gx, gy]
        action["boost"] = bool((energy > 45 and (dodge or outside_now or (urgent and d_next > 90)))
                               or (escaping and energy > 35))
        action["jump"] = bool(me.get("grounded", True) and escaping)

        # ---- aim and fire
        default_ranges = {"gun": 230, "laser": 280, "lightning": 45}
        reach = max((w["stats"].get("range", default_ranges[w["kind"]])
                     for w in weapons if w and w["kind"] in default_ranges), default=0)
        gun, second = weapons[0], weapons[1]
        tgt = None
        # Keep movement keyed to the closest mech. Aim at the closest one
        # with a clear firing lane; remembered targets can be behind walls.
        for contact in enemies:
            if distance(pos, contact["pos"]) >= reach:
                break
            height = 4.5 * (1 + 0.05 * contact.get("cubes", 0))
            target_pos = contact["pos"]
            if self.nav and not self.nav.shot_clear(
                (x, y, muzzle_z), (*target_pos[:2], target_pos[2] + height)
            ):
                continue
            tgt = (contact, height, True)
            break
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
                (second["kind"] == "laser" and dist < second["stats"].get("range", 280))
                or (second["kind"] == "lightning" and energy > 20
                    and dist < second["stats"].get("range", 45) - 2)))
            travel = 0.0
            if not instant:
                for _ in range(2):
                    travel = min(0.8, math.hypot(tp[0] + tv[0] * travel - x,
                                                 tp[1] + tv[1] * travel - y) / max(speed, 1.0))
            lx, ly = tp[0] + tv[0] * travel, tp[1] + tv[1] * travel
            lz = tp[2] + h + tv[2] * travel
            yaw = math.atan2(ly - y, lx - x)
            pitch = math.atan2(lz - muzzle_z, max(0.01, math.hypot(lx - x, ly - y)))
            action["aim"] = [yaw, pitch]
            yaw_err = wrap(yaw - me["aim"][0])
            pitch_err = pitch - me["aim"][1]
            aligned = (abs(yaw_err) < max(0.03, math.atan2(2.3, dist))
                       and abs(pitch_err) < max(0.04, math.atan2(3.5, dist)))
            clear = True
            if self.nav is not None and dist > 15:
                clear = self.nav.shot_clear((x, y, muzzle_z), (tp[0], tp[1], tp[2] + h))
            if gun and gun["kind"] == "gun":
                gun_range = gun["stats"].get("range", 230) if is_mech else 90
                action["fire"][0] = bool(aligned and clear and dist < gun_range and gun.get("ammo", 1) > 0)
                can_top_up = (is_mech and not clear and enemy_dist > 60 and not urgent
                              and tick - self.damage_tick > 40 and tick - self.listen_tick > 40
                              and gun.get("ammo", 1) <= gun["stats"].get("mag", 12) // 2)
                if (gun.get("ammo", 1) == 0 or can_top_up) and gun.get("reload", 0) <= 0:
                    action["reload"] = True
            if is_mech and second and second.get("cooldown", 0) <= 0 and clear:
                kind, st = second["kind"], second["stats"]
                if kind == "lightning":
                    near_aim = abs(yaw_err) < 0.2
                    action["fire"][1] = bool(near_aim and dist < st.get("range", 45) - 2 and energy > 20)
                elif kind == "laser" and dist < st.get("range", 280):
                    release_at = max(st.get("min_charge", 0), 0.72 * st.get("charge", 1))
                    # hold to charge; let go only when charged enough AND lined up
                    action["fire"][1] = not (second.get("charge", 0) >= release_at and aligned)
        else:
            if self.damage_bearing is not None and tick - self.damage_tick < 30:
                yaw = self.damage_bearing
            elif self.listen_bearing is not None and tick - self.listen_tick < 15:
                yaw = self.listen_bearing
            elif gx or gy:
                yaw = math.atan2(gy, gx)
            else:
                yaw = (tick * 0.025) % (2 * math.pi) - math.pi
            action["aim"] = [yaw, 0.0]
            if gun and gun["kind"] == "gun" and gun.get("ammo", 1) < gun["stats"].get("mag", 12):
                action["reload"] = True
        return action


mf.run(Bot)
