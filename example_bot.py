"""Darwin Survivor: a lightweight placement-focused Mech Royale bot."""
import math
import mechfight as mf


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(dx, dy):
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 0.001 else (0.0, 0.0)


class Bot:
    name = "Darwin Survivor"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.pickup_radius = cfg["loot"]["pickup_radius"]
        self.cube_radius = cfg.get("cubes", {}).get("pickup_radius", 3.5)
        self.last_position = None
        self.stuck = 0

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        pos = me["pos"]
        x, y = pos[:2]
        zone = obs["zone"]
        center = zone["center"]
        next_center = zone["next_center"]
        radius = zone["radius"]
        next_radius = zone["next_radius"]
        hp_ratio = me["hp"] / max(1, me["max_hp"])
        energy = me["energy"]
        weapons = me["weapons"]
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        enemies.sort(key=lambda m: distance(pos, m["pos"]))
        enemy = enemies[0] if enemies else None
        enemy_dist = distance(pos, enemy["pos"]) if enemy else float("inf")
        action: mf.Action = {"tick": obs["tick"], "fire": [False, False]}

        # Prefer durability and movement upgrades over damage.
        offer = obs.get("offer")
        if offer:
            choices = offer.get("choices", [])
            def priority(choice):
                s = (choice.get("id", "") + " " + choice.get("name", "") + " " + choice.get("desc", "")).lower()
                terms = [("armor", 100), ("repair", 95), ("regen", 90),
                         ("health", 85), ("hp", 85), ("speed", 70),
                         ("energy", 65), ("shield", 60), ("sensor", 35)]
                return max([score for word, score in terms if word in s] or [10])
            if choices:
                action["pick"] = max(range(len(choices)), key=lambda i: priority(choices[i]))

        # Grab a second weapon when in range; prefer lightning or laser.
        held = {w["kind"] for w in weapons if w}
        pickups = [p for p in obs["pickups"] if p["kind"] not in held]
        nearby = [p for p in pickups if distance(pos, p["pos"]) < self.pickup_radius - 0.2
                  and abs(pos[2] - p["pos"][2]) < 3.5]
        if nearby and weapons[1] is None:
            nearby.sort(key=lambda p: (-(3 if p["kind"] == "lightning" else 2 if p["kind"] == "laser" else 1), distance(pos, p["pos"])))
            action["take"] = nearby[0]["id"]

        # Stay in the next safe zone with a safety margin.
        safe_radius = max(8.0, next_radius - 25.0)
        d_next = distance(pos, next_center)
        outside = not me["in_zone"] or distance(pos, center) > radius - 18
        urgent_zone = outside or d_next > safe_radius
        goal = next_center
        if not urgent_zone and hp_ratio > 0.55 and enemy_dist > 75:
            # Collect only nearby loot, avoiding lengthy trips.
            candidates = []
            for cube in obs["cubes"]:
                d = distance(pos, cube["pos"])
                if d < 65 and distance(cube["pos"], next_center) < safe_radius:
                    candidates.append((d, cube["pos"]))
            if weapons[1] is None:
                for item in pickups:
                    d = distance(pos, item["pos"])
                    if d < 65 and distance(item["pos"], next_center) < safe_radius:
                        candidates.append((d + 8, item["pos"]))
            if candidates:
                goal = min(candidates, key=lambda c: c[0])[1]

        gx, gy = unit(goal[0] - x, goal[1] - y)
        if not urgent_zone and distance(pos, goal) < 12:
            gx, gy = 0.0, 0.0

        # Evade threats, but never retreat deeper outside the zone.
        if enemy and enemy_dist < (95 if hp_ratio < 0.6 else 55) and not outside:
            away_x, away_y = unit(x - enemy["pos"][0], y - enemy["pos"][1])
            weight = 0.85 if hp_ratio < 0.6 else 0.65
            gx, gy = unit((1 - weight) * gx + weight * away_x,
                          (1 - weight) * gy + weight * away_y)
            # If retreat would push us beyond the next zone, head toward its center.
            if distance((x + gx * 18, y + gy * 18), next_center) > max(5, next_radius - 10):
                gx, gy = unit(next_center[0] - x, next_center[1] - y)

        # Detect getting stuck on walls and try a perpendicular direction.
        if self.last_position is not None and math.hypot(x - self.last_position[0], y - self.last_position[1]) < 0.25 and (gx or gy):
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_position = (x, y)
        if self.stuck > 12:
            side = 1 if (obs["tick"] // 100) % 2 == 0 else -1
            gx, gy = unit(gx - side * gy, gy + side * gx)
        action["move"] = [gx, gy]
        action["boost"] = bool(energy > 45 and (outside or (urgent_zone and d_next > 90)))
        action["jump"] = bool(me["grounded"] and self.stuck > 22 and obs["tick"] % 12 == 0)

        # Aim with simple linear target leading, using the gun's projectile speed.
        if enemy and enemy_dist < 220:
            target = enemy["pos"]
            gun = weapons[0]
            speed = gun["stats"].get("speed", 252) if gun and gun["kind"] == "gun" else 252
            travel = min(0.8, enemy_dist / max(speed, 1))
            tx = target[0] + enemy["vel"][0] * travel
            ty = target[1] + enemy["vel"][1] * travel
            tz = target[2] + 4.5 + enemy["vel"][2] * travel
            yaw = math.atan2(ty - y, tx - x)
            pitch = math.atan2(tz - (pos[2] + 7), max(0.01, math.hypot(tx - x, ty - y)))
            action["aim"] = [yaw, pitch]
            if gun and gun["kind"] == "gun":
                action["fire"][0] = enemy_dist < gun["stats"].get("range", 230)
                if gun.get("ammo", 1) <= 2 and gun.get("reload", 0) <= 0:
                    action["reload"] = True
            second = weapons[1]
            if second and second.get("cooldown", 0) <= 0:
                kind = second["kind"]
                if kind == "lightning":
                    action["fire"][1] = enemy_dist < second["stats"].get("range", 45) - 2 and energy > 20
                elif kind == "laser":
                    # Hold to charge, release to shoot when adequately charged.
                    action["fire"][1] = enemy_dist < second["stats"].get("range", 280) and second.get("charge", 0) < 0.72
                elif kind == "mortar":
                    # Mortar ballistic aiming is not implemented; avoid wasting shells.
                    action["fire"][1] = False
        else:
            # Look toward movement, or scan when stationary.
            yaw = math.atan2(gy, gx) if gx or gy else (obs["tick"] * 0.025) % (2 * math.pi) - math.pi
            action["aim"] = [yaw, 0.0]
            if weapons[0] and weapons[0]["kind"] == "gun" and weapons[0].get("ammo", 1) < weapons[0]["stats"].get("mag", 12):
                action["reload"] = True
        return action


mf.run(Bot)
