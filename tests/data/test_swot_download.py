from pathlib import Path

from poseidon.data.sources.swot.download import SWOTDownloadConfig, SWOTQuery, run_swot_download


class StubEarthaccess:
    def __init__(self, records):
        self.records = records
        self.download_calls = []
        self.logged_in = False

    def login(self):
        self.logged_in = True

    def search_data(self, *, short_name, bounding_box, temporal):
        return list(self.records.get(short_name, []))

    def download(self, records, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for record in records:
            name = record["meta"]["native-id"]
            path = destination / name
            path.write_text("stub")
            paths.append(path)
        self.download_calls.append(paths)
        return paths


def test_run_swot_download_plans_replacement(tmp_path):
    outdir = tmp_path / "swot"
    outdir.mkdir()
    local_path = outdir / "SWOT_FAKE_20250101T000000_20250101T003000_DGAA_01.nc"
    local_path.write_text("old")

    query = SWOTQuery(
        product_key="lr",
        bbox=(-99.0, 17.0, -79.0, 31.0),
        start="2025-01-01",
        end="2025-01-02",
    )

    stub = StubEarthaccess(
        {
            "SWOT_L2_LR_SSH_Expert_2.0": [
                {"meta": {"native-id": "SWOT_FAKE_20250101T000000_20250101T003000_PGAB_01.nc"}},
                {"meta": {"native-id": "SWOT_FAKE_20250101T000000_20250101T003000_DGAA_01.nc"}},
            ]
        }
    )

    result = run_swot_download(query, outdir=outdir, dry_run=True, api=stub)

    assert result.plan.found == 2
    assert result.plan.best == 1
    assert [p.name for p in result.downloaded] == ["SWOT_FAKE_20250101T000000_20250101T003000_PGAB_01.nc"]
    assert [p.name for p in result.deleted] == [local_path.name]
    assert stub.download_calls == []


def test_config_from_mapping(tmp_path):
    cfg = {
        "product": "lr",
        "bbox": [-99.0, 17.0, -79.0, 31.0],
        "start": "2025-01-01",
        "end": "2025-01-05",
        "outdir": str(tmp_path / "swot"),
        "passes": [118, 120],
        "dry_run": True,
        "purge": True,
    }

    download_cfg = SWOTDownloadConfig.from_mapping(cfg)
    assert download_cfg.product == "lr"
    assert download_cfg.passes == ("118", "120")
    assert download_cfg.dry_run is True
    assert download_cfg.purge is True
    assert download_cfg.outdir == tmp_path / "swot"
    assert download_cfg.to_query() == SWOTQuery(
        product_key="lr",
        bbox=(-99.0, 17.0, -79.0, 31.0),
        start="2025-01-01",
        end="2025-01-05",
        passes=("118", "120"),
    )
