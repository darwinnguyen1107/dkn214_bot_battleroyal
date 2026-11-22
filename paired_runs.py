"""Paired placement comparison using a user-supplied local arena runner."""
import argparse
import json
import os
import statistics
import subprocess
import sys

FLAGS=("USE_NAV","SETTLE","STRAFE","FARM","DROPS")

def play(command,flags,seed):
    env=dict(os.environ)
    env.update({"ATLAS_"+k:str(value) for k,value in flags.items()})
    env["ATLAS_MATCH_SEED"]=str(seed)
    result=subprocess.run([s.replace("{seed}",str(seed)) for s in command],
                          env=env,capture_output=True,text=True,timeout=180,check=True)
    for line in reversed(result.stdout.splitlines()):
        try:msg=json.loads(line)
        except ValueError:continue
        if isinstance(msg,dict) and msg.get("type")=="end":
            place=msg["place"]
            if isinstance(place,int) and 1<=place<=16:return place
    raise ValueError("runner must emit an end message for the submitted player on stdout")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds",required=True,help="comma-separated integer seeds")
    ap.add_argument("--disable",nargs="+",choices=FLAGS,required=True)
    ap.add_argument("runner",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    command=args.runner[1:] if args.runner[:1]==["--"] else args.runner
    if not command:ap.error("supply a match runner after --")
    seeds=list(dict.fromkeys(int(v) for v in args.seeds.split(",")))
    base=dict.fromkeys(FLAGS,1);variant=dict(base,**dict.fromkeys(args.disable,0))
    pairs=[]
    for i,seed in enumerate(seeds):
        variants=[("A",base),("B",variant)]
        if i%2:variants.reverse()
        row={v:play(command,f,seed) for v,f in variants}
        pairs.append(row)
        print(json.dumps(dict(seed=seed,**row)),flush=True)
    change=[r["A"]-r["B"] for r in pairs]
    print(json.dumps({"pairs":len(pairs),"mean_place_A":statistics.mean(r["A"] for r in pairs),
                      "mean_place_B":statistics.mean(r["B"] for r in pairs),
                      "places_gained_by_B":statistics.mean(change),"B_wins":sum(d>0 for d in change),
                      "ties":change.count(0)}))

if __name__=="__main__":
    main()
