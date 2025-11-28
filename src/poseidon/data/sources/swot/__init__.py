from . import download
from . import preprocess
from . import shard_builder
from poseidon.data.sources.watermask import download as watermask_download

from typing import Mapping, Any
from poseidon.data.sources.base import DataSource, register_source


@register_source
class SWOTSource(DataSource):
    name = "swot"

    def download(self, cfg: Mapping[str, Any]) -> None:
        if hasattr(download, "main"):
            download.main(cfg)
        else:
            raise NotImplementedError("download.main not implemented for SWOT")

    def preprocess(self, cfg: Mapping[str, Any]) -> None:
        if hasattr(preprocess, "main"):
            preprocess.main(cfg)
        else:
            raise NotImplementedError("preprocess.main not implemented for SWOT")

    def build_shards(self, cfg: Mapping[str, Any]) -> None:
        if hasattr(shard_builder, "main"):
            shard_builder.main(cfg)
        else:
            raise NotImplementedError("shard_builder.main not implemented for SWOT")
