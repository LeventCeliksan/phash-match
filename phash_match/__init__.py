"""Perceptual image hashing and near-duplicate detection."""
from .core import (ImageHash, Match, average_hash, compare, difference_hash, find_duplicates, phash,
                   similarity)

__all__ = ["phash", "average_hash", "difference_hash", "compare", "find_duplicates", "similarity",
           "ImageHash", "Match"]
