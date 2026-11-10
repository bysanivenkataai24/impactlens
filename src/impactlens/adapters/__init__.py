"""Model runtimes. Import concrete adapters lazily: they pull in heavy libraries."""

from .base import AdapterInfo, ModelUnavailable, RawInstance, SegmentationAdapter

__all__ = ["AdapterInfo", "ModelUnavailable", "RawInstance", "SegmentationAdapter"]
