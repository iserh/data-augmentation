"""Generation algorithms."""
from . import augmentations  # noqa: F401
from .interpolation import Interpolation, interpolate_along_class, interpolate_along_dimension  # noqa: F401
from .noise import Noise  # noqa: F401
from .scripts import augment_dataset_using_per_class_vaes, augment_dataset_using_single_vae  # noqa: F401
