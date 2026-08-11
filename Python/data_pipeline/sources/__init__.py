from data_pipeline.sources.base import DiscoveryResult, SourceAdapter
from data_pipeline.sources.manifest import ManifestAdapter
from data_pipeline.sources.open_images import OpenImagesAdapter


def create_adapter(kind: str, settings: dict[str, object]) -> SourceAdapter:
    adapters = {"manifest": ManifestAdapter, "open_images": OpenImagesAdapter}
    try:
        adapter_type = adapters[kind]
    except KeyError as error:
        raise ValueError(
            f"unknown source adapter {kind!r}; available: {sorted(adapters)}"
        ) from error
    return adapter_type(settings)


__all__ = ["DiscoveryResult", "SourceAdapter", "create_adapter"]
