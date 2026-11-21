"""Build nav_cache.bin from the kit's map so the bot loads it instantly at match start.

Usage:
    python3 build_nav.py [path/to/arena]

Re-run this whenever navgrid.py changes (v8 adds the openness map used for hide spots).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from navgrid import NavGrid  # noqa: E402


def find_map_dir(cli_arg=None):
    candidates = [cli_arg] if cli_arg else [
        os.path.join(HERE, "arena"),
        os.path.join(HERE, "..", "arena"),
        os.path.join(HERE, "..", "mechfight-kit", "arena"),
        "/arena",
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "v2", "world.json")):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not find v2/world.json. Pass the arena folder path explicitly:\n"
        "  python3 build_nav.py path/to/arena"
    )


def main():
    map_dir = find_map_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    out = os.path.join(HERE, "nav_cache.bin")
    print("Building nav cache from:", map_dir)

    t0 = time.perf_counter()
    nav = NavGrid(map_dir, cache_path=None, budget_s=900.0)
    nav.save_cache(out)

    walls = sum(1 for v in nav.wblock if v == 1)
    margins = sum(1 for v in nav.wblock if v == 2)
    bridges = sum(1 for low in nav.low_base if 0.0 < low < 1e8)
    free = [nav.open[i] for i in range(len(nav.open)) if nav.wblock[i] == 0]
    print(
        "Built %dx%d grid | walls=%d, margins=%d, bridge_cells=%d | %.2fs -> %s (%d KB)"
        % (nav.gnx, nav.gny, walls, margins, bridges, time.perf_counter() - t0, out, os.path.getsize(out) // 1024)
    )
    if free:
        free.sort()
        print("Openness of free cells (0=enclosed, 255=wide open): min=%d p10=%d median=%d p90=%d max=%d" % (
            free[0], free[len(free) // 10], free[len(free) // 2], free[9 * len(free) // 10], free[-1]))

    again = NavGrid(map_dir, cache_path=out)
    assert again.from_cache, "cache did not load"
    assert again.tminx == nav.tminx and again.tminy == nav.tminy
    assert again.tstep == nav.tstep and again.tnx == nav.tnx and again.tny == nav.tny
    assert bytes(again.wblock) == bytes(nav.wblock)
    assert again.top.tobytes() == nav.top.tobytes()
    assert again.low_base.tobytes() == nav.low_base.tobytes()
    assert again.open is not None and bytes(again.open) == bytes(nav.open)
    print("Cache reload verification OK")


if __name__ == "__main__":
    main()
