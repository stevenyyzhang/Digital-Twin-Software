"""Illustrative radio-map layer for the no-model demo.

The values here are deliberately synthetic.  They are not Sionna results or
predictions from a trained model.  A later model adapter can provide the same
grid geometry and display values without changing the Polyscope rendering.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import polyscope as ps


LAYER_NAME = "Demo model radio map"


def make_illustrative_grid(
    bounds: tuple[float, float, float, float],
    tx_xy: tuple[float, float],
    shape: tuple[int, int] = (80, 120),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X/Y coordinates and illustrative display values in dBm."""
    xmin, xmax, ymin, ymax = bounds
    rows, cols = shape
    if not (xmin < xmax and ymin < ymax and rows >= 2 and cols >= 2):
        raise ValueError("Invalid radio-map bounds or grid shape")

    x, y = np.meshgrid(
        np.linspace(xmin, xmax, cols),
        np.linspace(ymin, ymax, rows),
    )
    dx = x - tx_xy[0]
    dy = y - tx_xy[1]
    distance = np.hypot(dx, dy)

    # Smooth lobes and a shadow-like region make the display easy to inspect.
    # This is a visual fixture, not a propagation equation.
    directional_lobe = 8.0 * np.exp(-((dy - 0.25 * dx) / 20.0) ** 2)
    shadow = 13.0 * np.exp(-((dx - 25.0) / 13.0) ** 2) * (dy < 0)
    values_dbm = -45.0 - 28.0 * np.log10(1.0 + distance / 8.0)
    values_dbm += directional_lobe - shadow
    return x, y, values_dbm


def register_illustrative_map(
    bounds: tuple[float, float, float, float],
    z: float,
    tx_xy: tuple[float, float],
    group,
    color_map: str = "turbo",
    visible: bool = False,
) -> tuple[float, float]:
    """Draw a translucent, scene-aligned synthetic coverage plane."""
    x, y, values_dbm = make_illustrative_grid(bounds, tx_xy)
    rows, cols = values_dbm.shape
    vertices = np.column_stack(
        (x.ravel(), y.ravel(), np.full(rows * cols, z))
    )
    indices = np.arange(rows * cols).reshape(rows, cols)
    faces = np.stack(
        (
            np.stack((indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1]), axis=-1),
            np.stack((indices[:-1, 1:], indices[1:, 1:], indices[1:, :-1]), axis=-1),
        ),
        axis=2,
    ).reshape(-1, 3)

    vmin, vmax = -100.0, -40.0
    normalized = np.clip((values_dbm.ravel() - vmin) / (vmax - vmin), 0, 1)
    colors = matplotlib.colormaps[color_map](normalized)[:, :3]

    mesh = ps.register_surface_mesh(LAYER_NAME, vertices, faces, enabled=visible)
    mesh.add_color_quantity("Illustrative strength (dBm)", colors, enabled=True)
    mesh.set_transparency(0.25)
    mesh.add_to_group(group)
    return vmin, vmax
