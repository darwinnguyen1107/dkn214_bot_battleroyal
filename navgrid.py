"""Navigation grid and line-of-sight helpers for the Mech Royale bot."""
import heapq
import json
import math
import os
import pickle
import sys
import time
from array import array

CACHE_VERSION = 2          # v1 caches (no openness map) still load; hide spots are then disabled
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

    def __init__(self, map_dir, cache_path=None, budget_s=10.0):
        t0 = time.perf_counter()
        deadline = t0 + budget_s
        cached = self._read_cache(cache_path) if cache_path else None
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
            self.open = self._compute_openness()

    @classmethod
    def _read_cache(cls, path):
        try:
            if array("f").itemsize != 4:
                return None
            with open(path, "rb") as f:
                d = pickle.load(f)
            if d.get("v") not in CACHE_ACCEPTED or d.get("cell") != cls.CELL:
                return None
            _, _, tstep, tnx, tny = d["meta"]
            n = int(math.ceil(tnx * tstep / cls.CELL)) * int(math.ceil(tny * tstep / cls.CELL))
            if len(d["top"]) != 4 * n or len(d["low"]) != 4 * n or len(d["wb"]) != n:
                return None
            return d
        except Exception:
            return None

    def save_cache(self, path):
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
        with open(path, "wb") as f:
            pickle.dump(d, f, protocol=4)

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
                    if (cx - cx0) % 32 == 0 and time.perf_counter() >= deadline:
                        raise TimeoutError("map too slow to rasterize")
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
            for cx in range(gnx):
                if cx % 128 == 0 and time.perf_counter() >= deadline:
                    raise TimeoutError("map too slow to inflate")
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
        Uses an O(8n) dynamic-programming sweep (distance-to-wall along each direction).
        """
        gnx, gny, w = self.gnx, self.gny, self.wblock
        n = gnx * gny
        total = [0] * n
        for dx, dy in DIRS8:
            run = [0] * n
            ys = range(gny - 1, -1, -1) if dy > 0 else range(gny)
            xs = list(range(gnx - 1, -1, -1)) if dx > 0 else list(range(gnx))
            for cy in ys:
                row = cy * gnx
                ny_ = cy + dy
                for cx in xs:
                    i = row + cx
                    if w[i] == 1:
                        continue
                    nx_ = cx + dx
                    if 0 <= nx_ < gnx and 0 <= ny_ < gny:
                        r = run[ny_ * gnx + nx_] + 1
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
        for k in [k for k, v in self.learned.items() if v < tick]:
            del self.learned[k]

    def learn_block(self, x, y, d, tick):
        for k in (2.5, 5.5):
            idx = self.cell_of(x + d[0] * k, y + d[1] * k)
            if idx is not None and len(self.learned) < 3000:
                self.learned[idx] = tick + 500

    def walk_clear(self, a, b, deadline=None, allow_wall_margin=False):
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(1, int(math.hypot(dx, dy) / 1.5))
        for k in range(1, n + 1):
            if deadline is not None and k % 16 == 1 and time.perf_counter() >= deadline:
                return False
            f = k / n
            idx = self.cell_of(a[0] + dx * f, a[1] + dy * f)
            if idx is None or idx in self.learned or self.wblock[idx] == 1:
                return False
            if self.wblock[idx] == 2 and not allow_wall_margin:
                return False
        return True

    def shot_clear(self, a, b):
        """3D line-of-sight check accounting for both terrain hills and building/bridge heights."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        horiz = math.hypot(dx, dy)
        n = int(horiz / 1.5)
        if n < 3:
            return True
        run = 0
        for k in range(2, n - 1):
            f = k / n
            px = a[0] + dx * f
            py = a[1] + dy * f
            ray_z = a[2] + dz * f
            if self.ground(px, py) > ray_z + 0.25:
                return False
            idx = self.cell_of(px, py)
            if idx is not None:
                b_low = self.low_base[idx]
                b_top = self.top[idx]
                if b_low - 0.3 <= ray_z <= b_top + 0.3:
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
                # Straight-line walkable, so the bot never takes a long exposed detour to reach it
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
