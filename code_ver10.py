"""Darwin Survivor v9: single-file placement-first Mech Royale bot.

Everything lives in this one file (navigation grid + bot + cache baking tool), so the kit's
conformance check can no longer fail with "No module named 'navgrid'".

What changed vs v8
  * navgrid.py and build_nav.py are merged in; nothing else needs to be committed.
  * Map cache can be BAKED INTO this file:   python3 darwin_survivor_v9.py --bake [path/to/arena]
    -> writes darwin_survivor_v9_baked.py. Submit that file. Instant start, no nav_cache.bin needed.
  * Without a baked/loose cache the grid is built in a low-priority background thread, so the bot
    reports "ready" immediately and plays with straight-line steering until the grid is finished.
  * Faster hot paths (walk_clear / shot_clear / ground), per-tick time tiers, cached line-of-sight.
  * Fixes: mortar dodge used a hard-coded gravity; upgrade picking matched noisy description text.
"""
import base64
import gc
import heapq
import json
import math
import os
import pickle
import random
import re
import sys
import threading
import time
import zlib
from array import array
from collections import deque

try:
    import mechfight as mf
except ImportError:  # allows `--bake` outside the kit folder
    mf = None

# Filled in by `--bake`. Leave this exact line alone.
_BAKED = None  # @@BAKED@@

# ---- Strategy toggles (flip one per submission to A/B test)
USE_NAV = True
SETTLE = True           # Fallback: hold a ring inside the next zone
HIDE_SPOTS = True       # Settle on a low-exposure spot between buildings
STRAFE = True
QUIET = True            # Sneak below step_speed when threats were heard or late game
AVOID_SOUND = True      # Drift away from nearby gunfire/explosions when nobody is visible
UNSEEN_EVADE = True     # Duck behind cover / sidestep when shot from outside our view
FARM_EARLY = True       # Crates/NPCs early for cubes and, above all, a second weapon
DROPS = False           # Supply drops draw every mech to one spot; flip on to A/B test
PLAN_BUDGET_S = 0.006
MAP_BUDGET_S = 10.0         # only used when loading from a cache
BG_BUILD_BUDGET_S = 240.0   # background build (no cache): gives up after this

# Per-tick time tiers in ms since the tick started (expensive work is skipped past these).
T_HOLD, T_PLAN, T_COVER, T_DODGE, T_SHOT = 10, 14, 14, 20, 24

ENGAGE_RANGE = 95.0
DROP_MIN_HP = 0.72
DROP_MIN_THREAT_DIST = 140.0
DROP_CLEAR_RADIUS = 95.0
KINETIC = ("gun", "mortar", "laser", "lightning")

BEATS = {"laser": "mortar", "mortar": "lightning", "lightning": "laser"}
WEAPON_RANK = {"lightning": 3, "laser": 2, "mortar": 1}


# =====================================================================================
# Navigation grid (formerly navgrid.py)
# =====================================================================================
CACHE_VERSION = 2
CACHE_ACCEPTED = (1, 2)
OPEN_CAP = 30              # openness rays are capped at 30 cells (90 m)

NEIGH = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
]
DIRS8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


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
    """Walkability, 3D line-of-sight and openness grid built from v2/world.json prisms and terrain."""
    CELL = 3.0

    def __init__(self, map_dir, cache_path=None, budget_s=10.0, cache_blob=None,
                 cache_only=False, polite=False, want_open=True):
        t0 = time.perf_counter()
        deadline = t0 + budget_s
        self.polite = polite            # background mode: nap regularly so act() stays fast
        self._last_nap = t0
        cached = None
        if cache_blob is not None:
            cached = self._parse_cache(cache_blob)
        elif cache_path:
            cached = self._read_cache(cache_path)
        if cached is None and cache_only:
            raise ValueError("no usable nav cache")

        world = None
        if cached is not None:
            self.tminx, self.tminy, self.tstep, self.tnx, self.tny = cached["meta"]
        else:
            with open(os.path.join(map_dir, "v2", "world.json")) as f:
                world = json.load(f)
            tr = world["terrain"]
            self.tminx = float(tr["minX"])
            self.tminy = float(tr["minY"])
            self.tstep = float(tr.get("step", 1.0))
            self.tnx = int(tr["nx"])
            self.tny = int(tr["ny"])

        self._load_terrain(map_dir)

        self.gnx = int(math.ceil(self.tnx * self.tstep / self.CELL))
        self.gny = int(math.ceil(self.tny * self.tstep / self.CELL))
        n = self.gnx * self.gny
        self.learned = {}
        self.from_cache = cached is not None
        self.open = None

        if cached is not None:
            self.top = array("f")
            self.top.frombytes(cached["top"])
            self.low_base = array("f")
            self.low_base.frombytes(cached["low"])
            self.wblock = bytearray(cached["wb"])
            op = cached.get("open")
            if op is not None and len(op) == n:
                self.open = bytearray(op)
        else:
            self.top = array("f", [-1e9]) * n
            self.low_base = array("f", [1e9]) * n
            self.wblock = bytearray(n)
            self._rasterize(world.get("prisms", []), deadline)
            del world
            self._inflate(deadline)
            if want_open:
                self.open = self._compute_openness()

    # ---- cache -------------------------------------------------------------------
    @classmethod
    def _parse_cache(cls, d):
        try:
            if array("f").itemsize != 4:
                return None
            if d.get("v") not in CACHE_ACCEPTED or d.get("cell") != cls.CELL:
                return None
            _, _, tstep, tnx, tny = d["meta"]
            n = int(math.ceil(tnx * tstep / cls.CELL)) * int(math.ceil(tny * tstep / cls.CELL))
            if len(d["top"]) != 4 * n or len(d["low"]) != 4 * n or len(d["wb"]) != n:
                return None
            return d
        except Exception:
            return None

    @classmethod
    def _read_cache(cls, path):
        try:
            with open(path, "rb") as f:
                return cls._parse_cache(pickle.load(f))
        except Exception:
            return None

    @staticmethod
    def decode_blob(b64):
        return pickle.loads(zlib.decompress(base64.b64decode(b64)))

    def cache_dict(self):
        d = {
            "v": CACHE_VERSION,
            "cell": self.CELL,
            "meta": (self.tminx, self.tminy, self.tstep, self.tnx, self.tny),
            "top": self.top.tobytes(),
            "low": self.low_base.tobytes(),
            "wb": bytes(self.wblock),
        }
        if self.open is not None:
            d["open"] = bytes(self.open)
        return d

    def save_cache(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.cache_dict(), f, protocol=4)

    def _pace(self):
        if self.polite:
            now = time.perf_counter()
            if now - self._last_nap > 0.004:
                time.sleep(0.008)
                self._last_nap = time.perf_counter()

    # ---- terrain / indexing --------------------------------------------------------
    def _load_terrain(self, map_dir):
        raw = array("h")
        with open(os.path.join(map_dir, "v2", "terrain.i16"), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder == "big":
            raw.byteswap()
        if len(raw) < self.tnx * self.tny:
            raise ValueError("terrain file too small")
        self.terrain = raw

    def ground(self, x, y):
        i = int((x - self.tminx) / self.tstep + 0.5)
        j = int((y - self.tminy) / self.tstep + 0.5)
        if i < 0:
            i = 0
        elif i >= self.tnx:
            i = self.tnx - 1
        if j < 0:
            j = 0
        elif j >= self.tny:
            j = self.tny - 1
        return self.terrain[j * self.tnx + i] * 0.01

    def cell_of(self, x, y):
        if x < self.tminx or y < self.tminy:
            return None
        cx = int((x - self.tminx) / self.CELL)
        cy = int((y - self.tminy) / self.CELL)
        if cx >= self.gnx or cy >= self.gny:
            return None
        return cy * self.gnx + cx

    def center(self, idx):
        cx, cy = idx % self.gnx, idx // self.gnx
        return (self.tminx + (cx + 0.5) * self.CELL, self.tminy + (cy + 0.5) * self.CELL)

    # ---- build -----------------------------------------------------------------------
    def _apply(self, idx, gx, gy, base, top):
        g = self.ground(gx, gy)
        if top > self.top[idx]:
            self.top[idx] = top
        if base < self.low_base[idx]:
            self.low_base[idx] = base
        if base < g + 9.0 and top > g + 1.2:
            self.wblock[idx] = 1

    def _rasterize(self, prisms, deadline):
        C = self.CELL
        for p in prisms:
            if time.perf_counter() >= deadline:
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
                    if (cx - cx0) % 32 == 0:
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("map too slow to rasterize")
                        self._pace()
                    px = self.tminx + (cx + 0.5) * C
                    if not point_in_poly(px, py, ring):
                        continue
                    if holes and any(point_in_poly(px, py, h) for h in holes):
                        continue
                    self._apply(cy * self.gnx + cx, px, py, base, top)
            for k in range(0, len(ring) - 1, 2):
                idx = self.cell_of(ring[k], ring[k + 1])
                if idx is not None:
                    self._apply(idx, ring[k], ring[k + 1], base, top)

    def _inflate(self, deadline):
        w, gnx, gny = self.wblock, self.gnx, self.gny
        src = bytes(w)
        for cy in range(gny):
            row = cy * gnx
            if time.perf_counter() >= deadline:
                raise TimeoutError("map too slow to inflate")
            self._pace()
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

    def _compute_openness(self):
        """Per-cell 0..255 score: how far you can see/shoot in 8 directions before a wall.

        Low = tucked between buildings (few firing lanes), high = wide open ground.
        O(8n) dynamic-programming sweep (distance-to-wall along each direction).
        """
        gnx, gny, w = self.gnx, self.gny, self.wblock
        n = gnx * gny
        total = [0] * n
        for dx, dy in DIRS8:
            run = [0] * n
            ys = range(gny - 1, -1, -1) if dy > 0 else range(gny)
            xs = list(range(gnx - 1, -1, -1)) if dx > 0 else list(range(gnx))
            for cy in ys:
                self._pace()
                row = cy * gnx
                ny_ = cy + dy
                row_ok = 0 <= ny_ < gny
                nrow = ny_ * gnx
                for cx in xs:
                    i = row + cx
                    if w[i] == 1:
                        continue
                    nx_ = cx + dx
                    if row_ok and 0 <= nx_ < gnx:
                        r = run[nrow + nx_] + 1
                        run[i] = r if r < OPEN_CAP else OPEN_CAP
                    else:
                        run[i] = 1
            for i in range(n):
                total[i] += run[i]
        denom = 8.0 * OPEN_CAP
        out = bytearray(n)
        for i in range(n):
            out[i] = 255 if w[i] == 1 else min(255, int(255.0 * total[i] / denom))
        return out

    # ---- queries -----------------------------------------------------------------------
    def pick_hold(self, pos, ncenter, safe_r, budget_s):
        """Pick a low-exposure walkable cell inside the next circle, biased toward us and the rim."""
        if self.open is None:
            return None
        deadline = time.perf_counter() + budget_s
        lim = max(10.0, 0.8 * safe_r)
        x, y = pos[0], pos[1]
        dx, dy = x - ncenter[0], y - ncenter[1]
        d = math.hypot(dx, dy)
        if d > lim:
            ax, ay = ncenter[0] + dx * lim / d, ncenter[1] + dy * lim / d
        else:
            ax, ay = x, y
        c0 = self.cell_of(ax, ay)
        if c0 is None:
            return None
        cx0, cy0 = c0 % self.gnx, c0 // self.gnx
        R = 24
        best, best_s = None, 1e18
        gnx, gny, wb, op, learned = self.gnx, self.gny, self.wblock, self.open, self.learned
        for cy in range(max(0, cy0 - R), min(gny - 1, cy0 + R) + 1):
            if time.perf_counter() >= deadline:
                break
            row = cy * gnx
            py = self.tminy + (cy + 0.5) * self.CELL
            for cx in range(max(0, cx0 - R), min(gnx - 1, cx0 + R) + 1):
                i = row + cx
                if wb[i] != 0 or i in learned:
                    continue
                px = self.tminx + (cx + 0.5) * self.CELL
                dc = math.hypot(px - ncenter[0], py - ncenter[1])
                if dc > lim:
                    continue
                s = op[i] + 0.6 * math.hypot(px - x, py - y) + 0.25 * (lim - dc)
                if s < best_s:
                    best_s, best = s, (px, py)
        return best

    def expire(self, tick):
        if self.learned:
            for k in [k for k, v in self.learned.items() if v < tick]:
                del self.learned[k]

    def learn_block(self, x, y, d, tick):
        for k in (2.5, 5.5):
            idx = self.cell_of(x + d[0] * k, y + d[1] * k)
            if idx is not None and len(self.learned) < 3000:
                self.learned[idx] = tick + 500

    def walk_clear(self, a, b, deadline=None, allow_wall_margin=False):
        ax, ay = a[0], a[1]
        dx, dy = b[0] - ax, b[1] - ay
        n = max(1, int(math.hypot(dx, dy) / 1.5))
        minx, miny, C, gnx, gny = self.tminx, self.tminy, self.CELL, self.gnx, self.gny
        wb, learned = self.wblock, self.learned
        inv = 1.0 / n
        for k in range(1, n + 1):
            if deadline is not None and (k & 15) == 1 and time.perf_counter() >= deadline:
                return False
            f = k * inv
            px = ax + dx * f
            py = ay + dy * f
            if px < minx or py < miny:
                return False
            cx = int((px - minx) / C)
            cy = int((py - miny) / C)
            if cx >= gnx or cy >= gny:
                return False
            idx = cy * gnx + cx
            w = wb[idx]
            if w == 1 or (learned and idx in learned):
                return False
            if w == 2 and not allow_wall_margin:
                return False
        return True

    def shot_clear(self, a, b):
        """3D line-of-sight check accounting for both terrain hills and building/bridge heights."""
        ax, ay, az = a[0], a[1], a[2]
        dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
        n = int(math.hypot(dx, dy) / 1.5)
        if n < 3:
            return True
        tminx, tminy, tstep = self.tminx, self.tminy, self.tstep
        tnx, tny, terr = self.tnx, self.tny, self.terrain
        C, gnx, gny = self.CELL, self.gnx, self.gny
        low, top = self.low_base, self.top
        inv = 1.0 / n
        run = 0
        for k in range(2, n - 1):
            f = k * inv
            px = ax + dx * f
            py = ay + dy * f
            ray_z = az + dz * f
            i = int((px - tminx) / tstep + 0.5)
            j = int((py - tminy) / tstep + 0.5)
            i = 0 if i < 0 else (tnx - 1 if i >= tnx else i)
            j = 0 if j < 0 else (tny - 1 if j >= tny else j)
            if terr[j * tnx + i] * 0.01 > ray_z + 0.25:
                return False
            if px < tminx or py < tminy:
                continue
            cx = int((px - tminx) / C)
            cy = int((py - tminy) / C)
            if cx >= gnx or cy >= gny:
                continue
            idx = cy * gnx + cx
            if low[idx] - 0.3 <= ray_z <= top[idx] + 0.3:
                run += 1
                if run >= 2:
                    return False
            else:
                run = 0
        return True

    def find_cover(self, pos, threat, center, safe_r, budget_s):
        """Walkable spot inside safe_r that a wall/hill hides from `threat`, reachable in a straight line."""
        deadline = time.perf_counter() + budget_s
        x, y = pos[0], pos[1]
        eye = (threat[0], threat[1], threat[2] + 8.0)
        base = math.atan2(y - threat[1], x - threat[0])
        order = (0, 2, -2, 3, -3, 1, -1, 4, -4, 5, -5)
        best = None
        best_score = 1e18
        for r in (10.0, 18.0, 28.0):
            for k in order:
                if time.perf_counter() >= deadline:
                    return best
                ang = base + k * 0.52
                cx, cy = x + r * math.cos(ang), y + r * math.sin(ang)
                if distance((cx, cy), center) > safe_r:
                    continue
                idx = self.cell_of(cx, cy)
                if idx is None or self.wblock[idx] == 1 or idx in self.learned:
                    continue
                if not self.walk_clear((x, y), (cx, cy), deadline, allow_wall_margin=True):
                    continue
                gz = self.ground(cx, cy)
                if not self.shot_clear(eye, (cx, cy, gz + 4.5)):
                    sc = r - 0.4 * distance((cx, cy), (threat[0], threat[1]))
                    if sc < best_score:
                        best_score = sc
                        best = (cx, cy)
            if best is not None:
                return best
        return best

    def nearest_open(self, idx):
        if self.wblock[idx] == 0 and idx not in self.learned:
            return idx
        gnx = self.gnx
        cx, cy = idx % gnx, idx // gnx
        margin_best, margin_best_d = None, 1e18
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
                    d2 = dx * dx + dy * dy
                    if self.wblock[j] == 0 and d2 < best_d:
                        best, best_d = j, d2
                    elif self.wblock[j] == 2 and d2 < margin_best_d:
                        margin_best, margin_best_d = j, d2
            if best is not None:
                return best
        return margin_best

    def plan(self, start, goal, budget_s):
        deadline = time.perf_counter() + budget_s
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
            if (n & 15) == 1 and time.perf_counter() >= deadline:
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
                if dx and dy:
                    sides = (cy * gnx + xx, yy * gnx + cx)
                    if any(wblock[cell] == 1 or cell in learned for cell in sides):
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
            while k + 1 < len(pts):
                if time.perf_counter() >= deadline:
                    return out + pts[i + 1:]
                if not self.walk_clear(pts[i], pts[k + 1], deadline, allow_wall_margin=True):
                    break
                k += 1
            out.append(pts[k])
            i = k
        return out


# =====================================================================================
# Bot
# =====================================================================================
def _log(*a):
    if mf is not None:
        try:
            mf.log(*a)
            return
        except Exception:  # noqa: BLE001
            pass
    print(*a, file=sys.stderr)


class Bot:
    name = "Darwin Survivor v9"

    def __init__(self, hello):
        cfg = hello["config"]
        self.my_slot = hello.get("slot", 0)
        self.rng = random.Random(hello.get("seed", 0))
        self.pickup_radius = cfg["loot"]["pickup_radius"]
        self.cube_radius = cfg.get("cubes", {}).get("pickup_radius", 3.5)
        self.step_speed = float(cfg.get("sound", {}).get("step_speed", 5.0))
        mortar_cfg = (cfg.get("weapons") or {}).get("mortar") or {}
        self.mortar_g = float(mortar_cfg.get("gravity", 22.0))
        self.mortar_v = float(mortar_cfg.get("speed", 72.0))
        self.upg_text = {}
        for u in (cfg.get("upgrades") or []):
            try:
                if isinstance(u, dict):
                    self.upg_text[u.get("id")] = json.dumps(u).lower()
            except Exception:  # noqa: BLE001
                pass
        self.t0 = time.perf_counter()
        self.slow_ticks = 0

        self.nav = None
        if USE_NAV and hello.get("map", {}).get("dir"):
            self._load_map(hello["map"]["dir"])

        self.hist = deque(maxlen=6)
        self.moving_obs = 0
        self.escape_until = -1
        self.escape_dir = (0.0, 0.0)
        self.hop_ledge = False

        self.path = []
        self.pi = 0
        self.path_goal = None
        self.plan_tick = -999

        self.damage_bearing = None
        self.damage_weapon = None
        self.damage_tick = -100
        self.listen_bearing = None
        self.listen_dist = None
        self.listen_tick = -100
        self.last_threat_yaw = None
        self.last_threat_tick = -100

        self.strafe_sign = 1
        self.strafe_until = 0
        self.laser_lost_los_ticks = 0

        self.cover = None
        self.cover_tick = -999
        self.cover_ctx = None

        self.hold_spot = None
        self.hold_key = None
        self.hold_tick = -999
        self.hold_banned = None
        self.hold_stuck = 0

        self.clear_cache = None

        gc.collect()
        gc.freeze()
        gc.set_threshold(50000, 20, 20)

    # ---- map loading: baked blob -> loose cache -> background build -------------------
    def _load_map(self, directory):
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            if _BAKED:
                try:
                    blob = NavGrid.decode_blob(_BAKED)
                    self.nav = NavGrid(directory, cache_blob=blob, cache_only=True)
                    _log("nav ready (baked, openness: %s)" % (self.nav.open is not None))
                    return
                except Exception as exc:  # noqa: BLE001
                    _log("baked cache unusable:", exc)
                    self.nav = None
            for path in (os.path.join(here, "nav_cache.bin"), os.path.join(os.getcwd(), "nav_cache.bin")):
                if os.path.isfile(path):
                    try:
                        self.nav = NavGrid(directory, cache_path=path, budget_s=MAP_BUDGET_S, cache_only=True)
                        _log("nav ready (nav_cache.bin, openness: %s)" % (self.nav.open is not None))
                        return
                    except Exception as exc:  # noqa: BLE001
                        _log("nav_cache.bin unusable:", exc)
                        self.nav = None
            # No cache anywhere: never block "ready" on a multi-second build.
            t = threading.Thread(target=self._bg_build, args=(directory,), daemon=True)
            t.start()
            _log("nav building in background")
        except Exception as exc:  # noqa: BLE001
            _log("nav disabled:", exc)
            self.nav = None

    def _bg_build(self, directory):
        try:
            try:
                sys.setswitchinterval(0.001)
            except Exception:  # noqa: BLE001
                pass
            nav = NavGrid(directory, budget_s=BG_BUILD_BUDGET_S, polite=True, want_open=False)
            self.nav = nav                       # usable immediately for pathing / line of sight
            _log("nav built in background")
            if HIDE_SPOTS:
                nav.open = nav._compute_openness()   # hide spots switch on once this finishes
                _log("openness ready")
            nav.polite = False
        except Exception as exc:  # noqa: BLE001
            _log("background nav failed:", exc)

    def spare(self, ms):
        return (time.perf_counter() - self.t0) * 1000.0 < ms

    def choose_upgrade(self, offer):
        choices = offer.get("choices", [])
        terms = [
            ("armor", 100), ("repair", 95), ("max_hp", 92), ("regen", 90), ("health", 88), ("hp", 88),
            ("walk_speed", 78), ("energy", 70), ("boost", 66), ("shield", 65), ("damage", 64),
            ("sensor", 60), ("cooldown", 58), ("rate", 55), ("speed", 50),
        ]

        def priority(c):
            cid = c.get("id") or ""
            head = (str(cid) + " " + str(c.get("name") or "")).lower()
            hit = max([sc for word, sc in terms if word in head] or [0])
            if hit:                                   # id/name is far more reliable than free text
                return hit
            tail = (str(c.get("desc") or "") + " " + self.upg_text.get(cid, "")).lower()
            return max([0.8 * sc for word, sc in terms if word in tail] or [10])
        return max(range(len(choices)), key=lambda i: priority(choices[i])) if choices else None

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

    def dodge(self, me, enemies, projectiles, center, safe_radius):
        """Sidestep incoming mortars or charging lasers."""
        nav = self.nav
        x, y, z = me["pos"]
        mx, my, mz = me["vel"]
        earliest = 1.4
        direction = None

        for e in enemies:
            if e.get("charging"):
                ex, ey, _ = e["pos"]
                eyaw = e.get("aim", [0, 0])[0]
                to_me = math.atan2(y - ey, x - ex)
                if abs(wrap(eyaw - to_me)) < 0.25 and distance((x, y), (ex, ey)) < 260:
                    sx, sy = unit(-(y - ey), x - ex)
                    for v in ((sx, sy), (-sx, -sy)):
                        end = (x + 10 * v[0], y + 10 * v[1])
                        if distance(end, center) <= safe_radius and (nav is None or nav.walk_clear((x, y), end)):
                            return v

        for shot in projectiles:
            if shot.get("kind") != "mortar":
                continue
            px, py, pz = shot["pos"]
            vx, vy, vz = shot["vel"]
            rvx, rvy = vx - mx, vy - my
            speed_sq = rvx * rvx + rvy * rvy
            if speed_sq < 1:
                continue
            t = ((x - px) * rvx + (y - py) * rvy) / speed_sq
            if not 0 < t <= earliest:
                continue
            dx = px - x + rvx * t
            dy = py - y + rvy * t
            pred_z = pz + vz * t - 0.5 * self.mortar_g * t * t
            if math.hypot(dx, dy) >= 9.0 or abs(pred_z - (z + 4.5)) > 8.0:
                continue
            sx, sy = unit(-rvy, rvx)
            safe = []
            for v in ((sx, sy), (-sx, -sy)):
                end = (x + 12 * v[0], y + 12 * v[1])
                if distance(end, center) <= safe_radius and (nav is None or nav.walk_clear((x, y), end)):
                    safe.append(v)
            if safe:
                direction = max(safe, key=lambda v: -(dx * v[0] + dy * v[1]))
                earliest = t
        return direction

    def steer(self, pos, goal, tick):
        x, y = pos[0], pos[1]
        direct = unit(goal[0] - x, goal[1] - y)
        nav = self.nav
        if nav is None:
            return direct
        try:
            if pos[2] > nav.ground(x, y) + 3.0:
                return direct
            if distance((x, y), goal) < 10 and nav.walk_clear((x, y), goal):
                return direct
            stale = (
                not self.path or self.path_goal is None
                or distance(self.path_goal, goal) > 15 or tick - self.plan_tick > 35
            )
            if stale and tick - self.plan_tick >= 4 and self.spare(T_PLAN):
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
            while j + 1 < len(p) and distance((x, y), p[j + 1]) < 40 and nav.walk_clear(
                (x, y), p[j + 1], allow_wall_margin=True
            ):
                j += 1
            self.pi = j
            if j == len(p) - 1 and distance((x, y), p[j]) < 1.0 and distance(p[j], goal) > 4.5:
                self.path = []
                return (0.0, 0.0)
            if distance((x, y), p[j]) < 1.0:
                return direct
            return unit(p[j][0] - x, p[j][1] - y)
        except Exception:
            return direct

    def select_target(self, pos, enemies, held):
        if not enemies:
            return None
        has_laser = "laser" in held

        def score(e):
            d = distance(pos, e["pos"])
            ehp = e.get("hp")
            max_ehp = max(1, e.get("max_hp", 1400))
            hp_frac = (ehp / max_ehp) if ehp is not None else 0.8
            s = d + hp_frac * 65.0
            if has_laser and "mortar" in e.get("weapons", []):
                s -= 45.0
            return s

        return min(enemies, key=score)

    def should_flee(self, me, enemy, enemies, hp_ratio, held):
        if len(enemies) >= 2:
            d0 = distance(me["pos"], enemies[0]["pos"])
            d1 = distance(me["pos"], enemies[1]["pos"])
            d_between = distance(enemies[0]["pos"], enemies[1]["pos"])
            if d0 < 120 and d1 < 120 and d_between < 85:
                return True

        if hp_ratio < 0.68:
            return True

        max_hp = max(1, me["max_hp"])
        e_hp = enemy.get("hp")
        if e_hp is None:
            e_hp = 0.8 * enemy.get("max_hp", max_hp)
        edge = (me["hp"] - e_hp) / max_hp + 0.04 * (me.get("cubes", 0) - enemy.get("cubes", 0))
        theirs = set(enemy.get("weapons", []))
        for k in held:
            if BEATS.get(k) in theirs:
                edge += 0.14
        for k in theirs:
            if BEATS.get(k) in held:
                edge -= 0.14
        others = sum(1 for m in enemies[1:] if distance(me["pos"], m["pos"]) < 95)
        edge -= 0.25 * others
        return edge < -0.08

    def act(self, obs):
        self.t0 = time.perf_counter()
        try:
            action = self._act(obs)
        except Exception as exc:  # noqa: BLE001
            print("act error: %r" % (exc,), file=sys.stderr)
            action = self._fallback(obs)
        ms = (time.perf_counter() - self.t0) * 1000.0
        if ms > 40.0:
            self.slow_ticks += 1
            if self.slow_ticks <= 20:
                print("slow tick %s: %.1f ms" % (obs.get("tick"), ms), file=sys.stderr)
        return action

    def _fallback(self, obs):
        action = {"tick": obs["tick"], "fire": [False, False]}
        try:
            me, zone = obs["self"], obs["zone"]
            c = zone.get("next_center") or zone["center"]
            action["move"] = list(unit(c[0] - me["pos"][0], c[1] - me["pos"][1]))
        except Exception:
            pass
        return action

    def _cover_for(self, ctx, pos, threat_pos, next_center, safe_edge, tick):
        """Cached cover lookup (recomputed every 4 ticks or when the threat context changes)."""
        if self.nav is None or not self.spare(T_COVER):
            return None
        if tick - self.cover_tick >= 4 or self.cover_ctx != ctx:
            self.cover = self.nav.find_cover(pos, threat_pos, next_center, safe_edge, 0.006)
            self.cover_tick = tick
            self.cover_ctx = ctx
        return self.cover

    def _shot_clear_cached(self, tick, a, b):
        """Line of sight, reused for one tick when neither end moved much (it is the priciest query)."""
        c = self.clear_cache
        if (
            c is not None and tick - c[0] <= 1
            and abs(a[0] - c[1]) < 2.5 and abs(a[1] - c[2]) < 2.5
            and abs(b[0] - c[3]) < 2.5 and abs(b[1] - c[4]) < 2.5
        ):
            return c[5]
        r = self.nav.shot_clear(a, b)
        self.clear_cache = (tick, a[0], a[1], b[0], b[1], r)
        return r

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
        next_radius = zone.get("next_radius") if zone.get("next_radius") is not None else radius

        hp_ratio = me["hp"] / max(1, me["max_hp"])
        energy = me["energy"]
        cubes = me.get("cubes", 0)
        weapons = me["weapons"]
        stats = me.get("stats", {})
        walk = max(1.0, stats.get("walk_speed", 11.7))
        muzzle_z = z + stats.get("muzzle_height", 7.0)
        held = {w["kind"] for w in weapons if w}
        my_id = me.get("id", self.my_slot)

        mechs = obs.get("mechs", [])
        enemies = sorted((m for m in mechs if m["kind"] == "mech"), key=lambda m: distance(pos, m["pos"]))
        nearest_enemy = enemies[0] if enemies else None
        nearest_enemy_dist = distance(pos, nearest_enemy["pos"]) if nearest_enemy else float("inf")
        npcs = [m for m in mechs if m["kind"] == "npc"]
        enemy = self.select_target(pos, enemies, held)
        enemy_dist = distance(pos, enemy["pos"]) if enemy else float("inf")
        action = {"tick": tick, "fire": [False, False]}

        for event in obs.get("events", []):
            if event.get("type") == "damaged" and event.get("bearing") is not None:
                self.damage_bearing = event["bearing"]
                self.damage_weapon = event.get("weapon")
                self.damage_tick = tick
                self.last_threat_yaw = event["bearing"]
                self.last_threat_tick = tick
        for s in obs.get("sounds", []):
            if (
                s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion")
                and s.get("dist") in ("near", "mid") and s.get("age", 0) < 0.5
            ):
                self.listen_bearing = s["bearing"]
                self.listen_dist = s.get("dist")
                self.listen_tick = tick

        if enemy:
            self.last_threat_yaw = math.atan2(enemy["pos"][1] - y, enemy["pos"][0] - x)
            self.last_threat_tick = tick

        provoked = tick - self.damage_tick < 40
        alive = obs.get("alive", 16)
        engage_range = 80.0 if alive > 8 else (130.0 if alive <= 5 else ENGAGE_RANGE)
        flee = bool(nearest_enemy and self.should_flee(me, nearest_enemy, enemies, hp_ratio, held))

        if nearest_enemy is not None and (
            nearest_enemy_dist < engage_range
            or (provoked and nearest_enemy_dist < 165.0)
            or (flee and nearest_enemy_dist < 150.0)
        ):
            enemy = nearest_enemy
            enemy_dist = nearest_enemy_dist

        offer = obs.get("offer")
        if offer:
            pick = self.choose_upgrade(offer)
            if pick is not None:
                action["pick"] = pick

        # Weapon looting & upgrading slot 1
        pickups = [p for p in obs.get("pickups", []) if p["kind"] not in held and abs(z - p["pos"][2]) < 3.5]
        nearby = [p for p in pickups if distance(pos, p["pos"]) < self.pickup_radius]
        if nearby:
            nearby.sort(key=lambda p: (-WEAPON_RANK.get(p["kind"], 0), distance(pos, p["pos"])))
            best_p = nearby[0]
            if weapons[1] is None:
                action["take"] = best_p["id"]
            elif not weapons[1].get("charge", 0):
                cur_rank = WEAPON_RANK.get(weapons[1]["kind"], 0)
                if WEAPON_RANK.get(best_p["kind"], 0) > cur_rank:
                    action["take"] = best_p["id"]
                    action["drop"] = 1

        # ---- Zone positioning
        d_next = distance(pos, next_center)
        safe_r = max(0.0, next_radius - min(25.0, next_radius * 0.5))
        outside_now = (not me.get("in_zone", True)) or distance(pos, center) > radius - min(12.0, radius * 0.5)
        need_t = max(0.0, d_next - safe_r) / walk
        tts = zone.get("shrink_start", 1e9) - t
        urgent = outside_now or (d_next > safe_r and (tts <= 0 or tts < need_t * 1.3 + 12))

        hold_r = max(0.0, 0.55 * next_radius - 10.0) if (SETTLE and next_radius > 60) else 0.0
        goal = None
        hold_goal = None
        if urgent:
            goal = next_center if d_next > safe_r or not outside_now else center
        else:
            use_spot = False
            nav = self.nav
            if HIDE_SPOTS and nav is not None and nav.open is not None and next_radius > 70:
                key = (round(next_center[0]), round(next_center[1]), round(next_radius))
                if key != self.hold_key:
                    self.hold_key, self.hold_spot, self.hold_tick, self.hold_stuck = key, None, -999, 0
                if (
                    self.hold_spot is None and self.hold_banned != key
                    and tick - self.hold_tick >= 30 and self.spare(T_HOLD)
                ):
                    self.hold_spot = nav.pick_hold(pos, next_center, safe_r, 0.006)
                    self.hold_tick = tick
                if self.hold_spot is not None and self.hold_banned != key:
                    use_spot = True
                    if distance((x, y), self.hold_spot) > 6.0:
                        goal = hold_goal = self.hold_spot
            if not use_spot and d_next > hold_r + 10:
                k = hold_r / d_next if d_next > 0.001 else 0.0
                goal = (next_center[0] + (x - next_center[0]) * k, next_center[1] + (y - next_center[1]) * k)

        # ---- Supply drops (off by default)
        pursuing = None
        if (
            DROPS and not urgent and hp_ratio >= DROP_MIN_HP
            and nearest_enemy_dist > DROP_MIN_THREAT_DIST
            and tick - self.listen_tick > 35
        ):
            best_time = float("inf")
            for dr in obs.get("drops", []):
                dd = distance(pos, dr["pos"])
                if dd > 140 or distance(dr["pos"], next_center) > safe_r or distance(dr["pos"], center) > radius:
                    continue
                if any(distance(e["pos"], dr["pos"]) < DROP_CLEAR_RADIUS for e in enemies):
                    continue
                if dr.get("holder") not in (None, my_id):
                    continue
                capture_radius = max(0.0, dr["radius"] - min(2.0, dr["radius"] * 0.25))
                travel = max(0.0, dd - capture_radius) / walk
                progress = dr["progress"] if dr["holder"] == my_id else 0.0
                finish = max(travel, dr["lands_in"]) + max(0.0, dr["capture_s"] - progress)
                if finish >= best_time:
                    continue
                if d_next > safe_r and finish + 2 >= tts:
                    continue
                best_time = finish
                pursuing = "drop"
                goal = None if dd <= capture_radius else (dr["pos"][0], dr["pos"][1])

        # ---- Loose cubes & weapon pickups (weapon-hungry until slot 1 is filled)
        need_weapon = weapons[1] is None
        if (
            pursuing is None and not urgent and hp_ratio > 0.55
            and nearest_enemy_dist > 65 and tick - self.listen_tick > 15
        ):
            cands = []
            for cube in obs.get("cubes", []):
                d = distance(pos, cube["pos"])
                if d < 75 and abs(z - cube["pos"][2]) < 3.5 and distance(cube["pos"], next_center) < safe_r:
                    cands.append((d, cube["pos"]))
            if need_weapon:
                reach_w = 130 if t < 120 else 95
                for item in pickups:
                    d = distance(pos, item["pos"])
                    if d < reach_w and distance(item["pos"], next_center) < safe_r:
                        cands.append((d * 0.8, item["pos"]))
            if cands:
                goal = min(cands, key=lambda c: c[0])[1][:2]
                pursuing = "loot"

        # ---- Early farming: crates (cube + 50% weapon drop) and, for cubes, NPCs
        farm = None
        quiet_area = (nearest_enemy_dist > 125) and (tick - self.listen_tick > 45)
        want_cubes = cubes < 3
        if (
            FARM_EARLY and (want_cubes or need_weapon) and t < (150.0 if need_weapon else 105.0)
            and pursuing is None and not urgent and hp_ratio > 0.80 and quiet_area
        ):
            best = None
            for c in obs.get("crates", []):
                d = distance(pos, c["pos"])
                if d < 55 and abs(z - c["pos"][2]) < 6 and (best is None or d < best[0]):
                    best = (d, c, 1.5)
            if want_cubes:
                for n in npcs:
                    d = distance(pos, n["pos"])
                    if d < 55 and abs(z - n["pos"][2]) < 6 and (best is None or d + 8.0 < best[0]):
                        best = (d + 8.0, n, 1.75)
            if best and distance(best[1]["pos"], next_center) < safe_r:
                farm = best
                pursuing = "farm"
                goal = (best[1]["pos"][0], best[1]["pos"][1]) if best[0] > 32 else None

        gx, gy = (0.0, 0.0) if goal is None else self.steer(pos, goal, tick)

        # ---- Fight / flee / strafe movement
        engaged = nearest_enemy is not None and (
            nearest_enemy_dist < engage_range or (provoked and nearest_enemy_dist < 165)
            or (flee and nearest_enemy_dist < 150)
        )
        mx, my = gx, gy
        safe_edge = max(5.0, next_radius - 10.0)
        if engaged and not urgent:
            threat = nearest_enemy
            threat_dist = nearest_enemy_dist
            ex, ey = threat["pos"][0], threat["pos"][1]
            ax, ay = unit(x - ex, y - ey)
            sx, sy = -ay, ax
            if tick >= self.strafe_until:
                self.strafe_sign = -self.strafe_sign
                self.strafe_until = tick + self.rng.randint(6, 14)
            sign = self.strafe_sign
            if self.nav is not None and not self.nav.walk_clear((x, y), (x + sx * sign * 10, y + sy * sign * 10)):
                sign = -sign
                self.strafe_sign = sign
            if flee:
                cover = self._cover_for("enemy", pos, threat["pos"], next_center, safe_r, tick)
                if cover is not None:
                    if distance((x, y), cover) < 3.0:
                        mx, my = 0.0, 0.0
                    else:
                        mx, my = unit(cover[0] - x, cover[1] - y)
                else:
                    mx, my = unit(0.80 * ax + 0.40 * sx * sign + 0.25 * gx,
                                  0.80 * ay + 0.40 * sy * sign + 0.25 * gy)
                    if distance((x + mx * 18, y + my * 18), next_center) > safe_edge:
                        mx, my = unit(next_center[0] - x, next_center[1] - y)
            elif STRAFE and threat_dist < 150:
                theirs = set(threat.get("weapons", []))
                if "lightning" in held and "mortar" not in theirs and energy > 35:
                    lo, hi = 22.0, 38.0
                elif "laser" in held or "mortar" in held:
                    lo, hi = 55.0, 120.0
                else:
                    lo, hi = 32.0, 105.0
                away_w = 0.65 if threat_dist < lo else (-0.4 if threat_dist > hi else 0.0)
                mx, my = unit(0.9 * sx * sign + away_w * ax + 0.15 * gx,
                              0.9 * sy * sign + away_w * ay + 0.15 * gy)
                if distance((x + mx * 12, y + my * 12), next_center) > safe_edge:
                    cx_, cy_ = unit(next_center[0] - x, next_center[1] - y)
                    mx, my = unit(mx + cx_, my + cy_)
        elif not urgent:
            # (a) Shot from outside our view: get out of the firing line instead of walking along it.
            if (
                UNSEEN_EVADE and provoked and self.damage_bearing is not None
                and self.damage_weapon in KINETIC
                and (nearest_enemy is None or nearest_enemy_dist > 165)
            ):
                b = self.damage_bearing
                if nearest_enemy is not None:
                    threat_pos = nearest_enemy["pos"]
                    ctx = "vis"
                else:
                    threat_pos = (x + 90.0 * math.cos(b), y + 90.0 * math.sin(b), z)
                    ctx = "phantom"
                cover = self._cover_for(ctx, pos, threat_pos, next_center, safe_r, tick)
                if cover is not None:
                    if distance((x, y), cover) < 3.0:
                        mx, my = 0.0, 0.0
                    else:
                        mx, my = unit(cover[0] - x, cover[1] - y)
                else:
                    ax, ay = -math.cos(b), -math.sin(b)
                    sx, sy = -ay, ax
                    sign = self.strafe_sign
                    mx, my = unit(0.6 * ax + 0.8 * sx * sign + 0.3 * gx, 0.6 * ay + 0.8 * sy * sign + 0.3 * gy)
                    if distance((x + mx * 18, y + my * 18), next_center) > safe_edge:
                        mx, my = unit(next_center[0] - x, next_center[1] - y)
            # (b) Nearby fight we are not part of: drift away and let them trade.
            elif (
                AVOID_SOUND and nearest_enemy is None and not provoked
                and tick - self.listen_tick < 25 and self.listen_bearing is not None
                and pursuing != "drop"
            ):
                ab = self.listen_bearing + math.pi
                w = 1.0 if self.listen_dist == "near" else 0.6
                vx, vy = unit(gx + w * math.cos(ab), gy + w * math.sin(ab))
                end = (x + vx * 15.0, y + vy * 15.0)
                if distance(end, next_center) <= safe_edge and (
                    self.nav is None or self.nav.walk_clear((x, y), (x + vx * 10.0, y + vy * 10.0))
                ):
                    mx, my = vx, vy
        gx, gy = mx, my

        # ---- Mortar & charging-laser dodge
        dodge = None
        if not urgent and (obs.get("projectiles") or enemies) and self.spare(T_DODGE):
            dodge = self.dodge(me, enemies, obs.get("projectiles", []), next_center, safe_r)
            if dodge:
                gx, gy = dodge

        # ---- Contextual silent walking (step sounds only carry ~40 m, so this matters near threats)
        q_speed = 0.9 * self.step_speed
        qf = min(1.0, q_speed / walk)
        near_threat_audio = (tick - self.listen_tick < 80) or (alive <= 6)
        quiet = False
        if (
            QUIET and near_threat_audio and enemy is None and not urgent and not outside_now
            and not dodge and pursuing not in ("drop", "loot") and tick >= self.escape_until
            and not provoked and (gx or gy)
            and not (tick - self.listen_tick < 25 and self.listen_dist == "near")
        ):
            quiet = d_next <= safe_r or (d_next - safe_r) / q_speed + 10.0 < tts

        # ---- Stuck detection + low-ledge hopping
        if math.hypot(gx, gy) > 0.3:
            self.moving_obs += 1
        else:
            self.moving_obs = 0
            self.hist.clear()
        self.hist.append((x, y))

        if self.moving_obs >= 5 and len(self.hist) >= 5 and tick >= self.escape_until:
            h = list(self.hist)
            net = distance(h[0], h[-1])
            if net < (0.6 if quiet else 1.2):
                d = unit(gx, gy)
                self.hop_ledge = False
                if self.nav is not None and energy > 55:
                    ahead_idx = self.nav.cell_of(x + d[0] * 3.5, y + d[1] * 3.5)
                    if ahead_idx is not None:
                        obs_top = self.nav.top[ahead_idx]
                        if z + 1.2 < obs_top <= z + 7.5:
                            self.hop_ledge = True

                if hold_goal is not None and goal is hold_goal:
                    self.hold_stuck += 1
                    if self.hold_stuck >= 3:      # hide spot seems unreachable: fall back to ring settling
                        self.hold_banned = self.hold_key
                        self.hold_spot = None

                if not self.hop_ledge:
                    if self.nav is not None and not dodge:
                        self.nav.learn_block(x, y, d, tick)
                    self.path = []
                    self.plan_tick = -999
                    ang = math.atan2(d[1], d[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(1.5, 2.3)
                    self.escape_dir = (math.cos(ang), math.sin(ang))
                else:
                    self.escape_dir = d

                self.escape_until = tick + (8 if self.hop_ledge else 10)
                self.hist.clear()
                self.moving_obs = 0

        escaping = tick < self.escape_until
        if escaping and dodge:
            self.escape_until = tick
            escaping = False
            self.hop_ledge = False
        elif escaping and not self.hop_ledge:
            dx, dy = self.escape_dir
            options = [(dx, dy), (-dx, -dy), (-dy, dx), (dy, -dx)]
            viable = []
            anchor = goal if (urgent and goal) else next_center
            for v in options:
                end = (x + 6 * v[0], y + 6 * v[1])
                if self.nav and not self.nav.walk_clear((x, y), end, allow_wall_margin=True):
                    continue
                if distance(end, anchor) > max(distance(pos, anchor), 6.0 if urgent else safe_r):
                    continue
                viable.append(v)
            if viable:
                gx, gy = min(viable, key=lambda v: distance((x + 6 * v[0], y + 6 * v[1]), anchor)) if urgent else viable[0]
            else:
                escaping = False
                self.escape_until = tick

        if quiet and not escaping:
            action["move"] = [gx * qf, gy * qf]
        else:
            action["move"] = [gx, gy]
        action["boost"] = bool(
            (energy > 45 and (dodge or outside_now or (urgent and d_next > 90)))
            or (escaping and self.hop_ledge and energy > 40)
        )
        action["jump"] = bool(me.get("grounded", True) and escaping)

        # ---- Aim & fire
        default_ranges = {"gun": 230, "laser": 280, "lightning": 45, "mortar": 220}
        reach = max(
            (w["stats"].get("range", default_ranges.get(w["kind"], 200)) for w in weapons if w),
            default=0
        )
        gun, second = weapons[0], weapons[1]
        tgt = None
        if enemy and enemy_dist < reach:
            tgt = (enemy, 4.5 * (1 + 0.05 * enemy.get("cubes", 0)), True)
        elif farm and farm[0] < 90:
            tgt = (farm[1], farm[2], False)

        hold_fire_for_regen = (
            bool(tgt and tgt[2]) and hp_ratio < 0.65
            and nearest_enemy_dist > 50 and not provoked
        )

        if tgt:
            obj, h, is_mech = tgt
            tp = obj["pos"]
            tv = obj.get("vel", (0.0, 0.0, 0.0))
            dist = distance(pos, tp)
            may_fire = (engaged or not is_mech) and not hold_fire_for_regen

            speed = gun["stats"].get("speed", 252) if gun and gun["kind"] == "gun" else 252
            instant = bool(
                is_mech and second and second.get("cooldown", 0) <= 0 and (
                    (second["kind"] == "laser" and dist < second["stats"].get("range", 280))
                    or (second["kind"] == "lightning" and energy > 20 and dist < second["stats"].get("range", 45) - 2)
                )
            )
            travel = 0.0
            if not instant:
                for _ in range(2):
                    travel = min(0.8, math.hypot(tp[0] + tv[0] * travel - x, tp[1] + tv[1] * travel - y) / max(speed, 1.0))

            lx, ly = tp[0] + tv[0] * travel, tp[1] + tv[1] * travel
            lz = tp[2] + h + tv[2] * travel
            yaw = math.atan2(ly - y, lx - x)
            pitch = math.atan2(lz - muzzle_z, max(0.01, math.hypot(lx - x, ly - y)))
            action["aim"] = [yaw, pitch]

            yaw_err = wrap(yaw - me["aim"][0])
            pitch_err = pitch - me["aim"][1]
            aligned = (
                abs(yaw_err) < max(0.04, math.atan2(3.2, dist))
                and abs(pitch_err) < max(0.055, math.atan2(4.2, dist))
            )

            clear = True
            # Only worth the ray-march when we could actually fire this tick.
            if self.nav is not None and dist > 12 and may_fire and self.spare(T_SHOT):
                clear = self._shot_clear_cached(tick, (x, y, muzzle_z), (tp[0], tp[1], tp[2] + h))

            if gun and gun["kind"] == "gun":
                gun_range = gun["stats"].get("range", 230) if is_mech else 90
                can_shoot = may_fire and aligned and clear and dist < gun_range and gun.get("ammo", 1) > 0
                action["fire"][0] = bool(can_shoot)
                if gun.get("ammo", 1) == 0 and gun.get("reload", 0) <= 0:
                    action["reload"] = True

            if is_mech and may_fire and second and second.get("cooldown", 0) <= 0:
                kind, st = second["kind"], second["stats"]
                if kind == "lightning" and clear:
                    near_aim = abs(yaw_err) < 0.24
                    action["fire"][1] = bool(near_aim and dist < st.get("range", 45) - 2 and energy > 22)
                elif kind == "laser" and dist < st.get("range", 280):
                    cur_charge = second.get("charge", 0)
                    max_charge = st.get("charge", 0.75)
                    release_at = max(st.get("min_charge", 0.15), 0.70 * max_charge)
                    laser_aligned = abs(yaw_err) < max(0.06, math.atan2(3.6, dist))
                    if cur_charge == 0:
                        if clear and abs(yaw_err) < 0.25:
                            action["fire"][1] = True
                            self.laser_lost_los_ticks = 0
                    else:
                        if clear:
                            self.laser_lost_los_ticks = 0
                            ready_to_fire = (cur_charge >= release_at and laser_aligned) or (cur_charge >= max_charge * 0.95)
                            action["fire"][1] = not ready_to_fire
                        else:
                            self.laser_lost_los_ticks += 1
                            action["fire"][1] = self.laser_lost_los_ticks <= 2
                elif kind == "mortar" and 25 < dist < 210 and clear:
                    m_speed = float(st.get("speed", self.mortar_v))
                    sol = self.lob(x, y, muzzle_z + 2.5, (tp[0], tp[1], tp[2] + h), tv, m_speed, self.mortar_g)
                    if sol is not None:
                        m_yaw, m_pitch, _ = sol
                        action["aim"] = [m_yaw, m_pitch]
                        if abs(wrap(m_yaw - me["aim"][0])) < 0.10 and abs(m_pitch - me["aim"][1]) < 0.07:
                            action["fire"][1] = True
        else:
            if self.damage_bearing is not None and tick - self.damage_tick < 40:
                yaw = self.damage_bearing
            elif self.last_threat_yaw is not None and tick - self.last_threat_tick < 40:
                yaw = self.last_threat_yaw
            elif self.listen_bearing is not None and tick - self.listen_tick < 20:
                yaw = self.listen_bearing
            elif gx or gy:
                yaw = math.atan2(gy, gx) + 1.3 * math.sin(tick * 0.07)
            else:
                yaw = (tick * 0.03) % (2 * math.pi) - math.pi
            action["aim"] = [wrap(yaw), 0.0]
            if gun and gun["kind"] == "gun" and gun.get("ammo", 1) < gun["stats"].get("mag", 12):
                action["reload"] = True

        return action


# =====================================================================================
# Cache baking (replaces build_nav.py):  python3 darwin_survivor_v9.py --bake [arena]
# =====================================================================================
def _find_map_dir(cli_arg=None):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [cli_arg] if cli_arg else [
        os.path.join(here, "arena"),
        os.path.join(here, "..", "arena"),
        os.path.join(here, "..", "mechfight-kit", "arena"),
        "/arena",
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "v2", "world.json")):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not find v2/world.json. Pass the arena folder path explicitly:\n"
        "  python3 darwin_survivor_v9.py --bake path/to/arena"
    )


def _bake(arena_arg=None):
    here = os.path.abspath(__file__)
    map_dir = _find_map_dir(arena_arg)
    print("Building nav grid from:", map_dir)
    t0 = time.perf_counter()
    nav = NavGrid(map_dir, budget_s=1800.0)
    print("Built %dx%d grid in %.1fs" % (nav.gnx, nav.gny, time.perf_counter() - t0))

    blob = base64.b64encode(zlib.compress(pickle.dumps(nav.cache_dict(), protocol=4), 9)).decode("ascii")
    with open(here, "r", encoding="utf-8") as f:
        src = f.read()
    pattern = re.compile(r"^_BAKED = .*# @@BAKED@@$", re.M)
    if not pattern.search(src):
        raise RuntimeError("marker line '_BAKED = ...  # @@BAKED@@' not found in source")
    baked = pattern.sub(lambda m: '_BAKED = "%s"  # @@BAKED@@' % blob, src, count=1)
    out = os.path.join(os.path.dirname(here), "darwin_survivor_v9_baked.py")
    with open(out, "w", encoding="utf-8") as f:
        f.write(baked)

    again = NavGrid(map_dir, cache_blob=NavGrid.decode_blob(blob), cache_only=True)
    assert again.from_cache and bytes(again.wblock) == bytes(nav.wblock)
    assert again.top.tobytes() == nav.top.tobytes()
    assert again.low_base.tobytes() == nav.low_base.tobytes()
    assert again.open is not None and bytes(again.open) == bytes(nav.open)
    print("Wrote %s (%d KB). Submit THAT file." % (out, os.path.getsize(out) // 1024))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--bake":
    _bake(sys.argv[2] if len(sys.argv) > 2 else None)
else:
    mf.run(Bot)
