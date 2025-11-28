                      
                                                                                                                 

import argparse, json, re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import earthaccess

S6_LR_NTC = [
    "JASON_CS_S6A_L2_ALT_LR_RED_OST_NTC_F09",
    "JASON_CS_S6A_L2_ALT_LR_RED_OST_NTC_F08",
]

S6_HR_NTC = [
    "JASON_CS_S6A_L2_ALT_HR_RED_OST_NTC_F09",
    "JASON_CS_S6A_L2_ALT_HR_RED_OST_NTC_F08",
]

PRODUCTS = {"lr": S6_LR_NTC, "hr": S6_HR_NTC}

GFNC = re.compile(r'_(\d{3})_(\d{3})_(\d{8}T\d{6})_(\d{8}T\d{6})_F(\d{2})', re.IGNORECASE)

def parse_name(name: str):
    m = GFNC.search(name)
    if not m: return None
    cyc, pas, t0, t1, fxx = m.groups()
    return {"cycle": int(cyc), "pass": int(pas), "t0": t0, "t1": t1, "baseline": int(fxx)}

def load_config(p: Path) -> dict:
    cfg = json.loads(p.read_text())
    need = ["min_lon","min_lat","max_lon","max_lat","start_date","end_date"]
    miss = [k for k in need if k not in cfg]
    if miss: raise ValueError(f"config missing keys: {miss}")
    return cfg

def search_all(short_names: List[str], bbox: Tuple[float,float,float,float], temporal: Tuple[str,str]) -> List[Dict]:
    out: List[Dict] = []
    for sn in short_names:
        rs = earthaccess.search_data(short_name=sn, bounding_box=bbox, temporal=temporal)
        for r in rs:
            r["_src_sn"] = sn
            r["_name"] = r.get("meta", {}).get("native-id", "")
            r["_p"] = parse_name(r["_name"])
            if r["_p"]:
                out.append(r)
    return out

def best_per_start(results: List[Dict], pref_order: List[str]) -> List[Dict]:
    """Group by t0; rank by baseline Fxx, then short-name order, then name."""
    order = {sn: i for i, sn in enumerate(pref_order)}
    buckets = defaultdict(list)
    for r in results:
        buckets[r["_p"]["t0"]].append(r)
    winners = []
    for t0, items in buckets.items():
        items.sort(key=lambda r: (r["_p"]["baseline"], -(1000 - order.get(r["_src_sn"], 999)), r["_name"]), reverse=True)
        winners.append(items[0])
    return winners

def reconcile_strict(outdir: Path, winners: List[Dict]) -> Tuple[List[Dict], List[Path]]:
    """
    Strict replacement:
      - If a local file with same t0 exists and has >= baseline, do nothing.
      - If winner baseline > best local baseline for that t0, download winner and delete lower-baseline locals for that t0.
      - If nothing local for t0, download if not already present by exact filename.
    """
    local_by_t0: dict[str, List[Tuple[Path, Dict]]] = defaultdict(list)
    for f in outdir.iterdir():
        if not f.is_file(): continue
        p = parse_name(f.name)
        if not p: continue
        local_by_t0[p["t0"]].append((f, p))

    to_dl: List[Dict] = []
    to_rm: List[Path] = []

    for r in winners:
        name = r["_name"]; p = r["_p"]; t0 = p["t0"]; b_remote = p["baseline"]
        local_set = local_by_t0.get(t0, [])
        if not local_set:
            if not (outdir / name).exists():
                to_dl.append(r)
            continue
        b_local_best = max(lp["baseline"] for _, lp in local_set)
        if b_remote > b_local_best:
            if not (outdir / name).exists():
                to_dl.append(r)
            for f, lp in local_set:
                if lp["baseline"] < b_remote and f.name != name:
                    to_rm.append(f)
        else:
            pass

    return to_dl, to_rm

def main():
    ap = argparse.ArgumentParser(description="Sentinel-6A L2 ALT NTC downloader (LR/HR), F09>F08, strict replace.")
    ap.add_argument("--product", choices=["lr","hr"], required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--passes", type=int, nargs="*", default=None, help="Optional pass filter, e.g. 215 216")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MINLON","MINLAT","MAXLON","MAXLAT"), default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bbox = tuple(args.bbox) if args.bbox else (cfg["min_lon"], cfg["min_lat"], cfg["max_lon"], cfg["max_lat"])
    temporal = (args.start or cfg["start_date"], args.end or cfg["end_date"])
    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    earthaccess.login()

    pref = PRODUCTS[args.product]
    found = search_all(pref, bbox, temporal)

    if args.passes:
        needles = [f"_{p:03d}_" for p in args.passes]
        found = [r for r in found if any(n in r["_name"] for n in needles)]

    winners = best_per_start(found, pref)
    to_dl, to_rm = reconcile_strict(outdir, winners)

    print(f"Found: {len(found)} | winners: {len(winners)} | replace: {len(to_rm)} | download: {len(to_dl)}")

    if args.dry_run:
        for f in to_rm: print("DRY-DELETE:", f.name)
        for r in to_dl: print("DRY-DOWNLOAD:", r["_name"])
        return

    for f in to_rm:
        try: f.unlink()
        except Exception as e:
            print(f"Warn: delete {f.name} failed: {e}")
    if to_dl:
        earthaccess.download(to_dl, str(outdir))
    print("Done.")

if __name__ == "__main__":
    main()
