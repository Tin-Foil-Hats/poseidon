from abc import ABC, abstractmethod
from typing import Mapping, Any, Dict, Type


class DataSource(ABC):
    name: str

    @abstractmethod
    def download(self, cfg: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, cfg: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_shards(self, cfg: Mapping[str, Any]) -> None:
        raise NotImplementedError


_REGISTRY: Dict[str, Type[DataSource]] = {}


def register_source(cls: Type[DataSource]) -> Type[DataSource]:
    if not hasattr(cls, "name") or not isinstance(cls.name, str):
        raise ValueError("DataSource subclasses must define a string name")
    _REGISTRY[cls.name] = cls
    return cls


def get_source(name: str) -> DataSource:
    if name not in _REGISTRY:
        raise KeyError(f"unknown data source {name}")
    cls = _REGISTRY[name]
    return cls()
