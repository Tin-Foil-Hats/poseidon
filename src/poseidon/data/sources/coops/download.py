import argparse
import json
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
from pathlib import Path

class NOAAStationFetcher:
    def __init__(
        self,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        min_age_years=None,
        start_date=None,
        min_duration_years=None,
        min_duration_months=None,
    ):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        self.min_age_years = min_age_years
        self.start_date = datetime.fromisoformat(start_date) if start_date else None
        self.min_duration_years = min_duration_years
        self.min_duration_months = min_duration_months
        self.base_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
        self.filtered_stations = []

    def fetch_stations(self):
        url = f"{self.base_url}/stations.json?status=active&type=waterlevels&bbox={self.bbox}"
        r = requests.get(url)
        r.raise_for_status()
        return r.json().get("stations", [])

    def filter_by_bbox(self, stations):
        return [
            s for s in stations
            if self.min_lon <= s["lng"] <= self.max_lon
            and self.min_lat <= s["lat"] <= self.max_lat
        ]

    def _get_station_details(self, station_id):
        url = f"{self.base_url}/stations/{station_id}/details.json"
        try:
            r = requests.get(url)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _station_meets_criteria(self, established_iso, disestablished_iso=None):
        try:
            established_dt = datetime.fromisoformat(established_iso)
        except Exception:
            return False, None
        disestablished_dt = None
        if disestablished_iso:
            try:
                disestablished_dt = datetime.fromisoformat(disestablished_iso)
            except Exception:
                pass
        if self.start_date:
            if established_dt >= self.start_date:
                return False, established_dt
            if disestablished_dt:
                duration_days = (disestablished_dt - self.start_date).days
            else:
                duration_days = (datetime.now() - self.start_date).days
            required_days = 0
            if self.min_duration_years:
                required_days += self.min_duration_years * 365
            if self.min_duration_months:
                required_days += self.min_duration_months * 30
            if duration_days < required_days:
                return False, established_dt
            return True, established_dt
        if self.min_age_years:
            age_days = (datetime.now() - established_dt).days
            return age_days >= 365 * self.min_age_years, established_dt
        return True, established_dt

    def run(self):
        print("\nFetching NOAA stations\n")
        with tqdm(total=1, desc="Fetching station list", ncols=80) as pbar:
            stations = self.fetch_stations()
            time.sleep(0.1)
            pbar.update(1)
        print(f"DONE : Fetched {len(stations)} stations.")
        with tqdm(total=1, desc="Filtering by bounding box", ncols=80) as pbar:
            bbox_filtered = self.filter_by_bbox(stations)
            time.sleep(0.1)
            pbar.update(1)
        print(f"DONE : {len(bbox_filtered)} stations within bounding box.")
        filtered = []
        for s in tqdm(bbox_filtered, desc="Checking station histories", ncols=80):
            details = self._get_station_details(s["id"])
            if not details:
                continue
            est = details.get("established")
            disest = details.get("disestablished")
            if not est:
                continue
            meets, est_dt = self._station_meets_criteria(est, disest)
            if meets:
                s["established"] = est_dt.strftime("%Y-%m-%d")
                s["years"] = round((datetime.now() - est_dt).days / 365.25, 1)
                filtered.append(s)
        self.filtered_stations = filtered
        print(f"\nDONE: Found {len(self.filtered_stations)} matching stations.\n")

    def to_dataframe(self):
        if not self.filtered_stations:
            print("No stations found. Run .run() first.")
            return pd.DataFrame()
        df = pd.DataFrame([
            {
                "ID": s["id"],
                "Name": s["name"],
                "Established": s["established"],
                "Years": s["years"]
            }
            for s in self.filtered_stations
        ])
        return df

    def save_txt(self, filepath):
        df = self.to_dataframe()
        if df.empty:
            print("No data to save.")
            return
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, sep=",", index=False, header=True)
        print(f"Saved {len(df)} stations to {p}")

    def display(self):
        for s in self.filtered_stations:
            print(f"{s['id']} | {s['name']} | {s['established']} | {s['years']} years")
        print(f"Total: {len(self.filtered_stations)} stations found.")


def main():
    parser = argparse.ArgumentParser(description="Fetch NOAA stations from JSON config or CLI args.")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--min_lon", type=float)
    parser.add_argument("--min_lat", type=float)
    parser.add_argument("--max_lon", type=float)
    parser.add_argument("--max_lat", type=float)
    parser.add_argument("--start_date", type=str)
    parser.add_argument("--min_age_years", type=int)
    parser.add_argument("--min_duration_years", type=int)
    parser.add_argument("--min_duration_months", type=int)
    parser.add_argument("--region_name", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        config = vars(args)

    region_name = config.get("region_name", "region").strip()
    safe_name = region_name.lower().replace(" ", "_")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_name = config.get("output", f"{safe_name}_stations.txt")
    output_file = outdir / output_name

    fetcher = NOAAStationFetcher(
        min_lon=config["min_lon"],
        min_lat=config["min_lat"],
        max_lon=config["max_lon"],
        max_lat=config["max_lat"],
        min_age_years=config.get("min_age_years"),
        start_date=config.get("start_date"),
        min_duration_years=config.get("min_duration_years"),
        min_duration_months=config.get("min_duration_months")
    )

    print(f"\nProcessing region: {region_name}")
    print(f"Output file: {output_file}\n")

    fetcher.run()
    if args.verbose:
        fetcher.display()
    fetcher.save_txt(str(output_file))

if __name__ == "__main__":
    main()
