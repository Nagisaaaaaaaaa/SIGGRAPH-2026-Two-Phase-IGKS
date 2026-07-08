import matplotlib.pyplot as plt
import numpy as np


def colormap(name: str, volume: np.ndarray) -> np.ndarray:
  if not hasattr(colormap, "cache"):
    colormap.cache = {}
  cache = colormap.cache

  dtype = volume.dtype if np.issubdtype(volume.dtype, np.floating) else np.dtype(np.float32)

  key = (name, dtype)
  if key not in cache:
    cache[key] = np.asarray(plt.get_cmap(name)(np.linspace(0.0, 1.0, 256, dtype=dtype))[:, :3], dtype=dtype)
  colors = cache[key]

  value255 = np.clip(volume.astype(dtype, copy=False) * dtype.type(255.0), dtype.type(0.0), dtype.type(255.0))
  i = np.minimum(value255.astype(np.uint32), 254)
  t = (value255 - i.astype(dtype))[..., None]
  return colors[i] + t * (colors[i + 1] - colors[i])
