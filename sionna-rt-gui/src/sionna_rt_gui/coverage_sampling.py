"""Seeded area-uniform directions for independent coverage batches.

Sionna RT 1.2.2's mesh solver seeds interaction randomness but always launches
the same Fibonacci lattice when samples_per_tx is unchanged. Repeated batches
therefore repeat LOS/specular hits. Adapt its lattice hook for this synchronous
solve only; restore the upstream function even when a solve fails.
"""
from contextlib import contextmanager
import importlib
import numpy as np
import mitsuba as mi


def stratified_square(count,seed):
    rng=np.random.default_rng(seed)
    points=rng.random((2,count),dtype=np.float32)
    # Sphere warp has constant area PDF on this square. Stratify one coordinate
    # and independently jitter the other, avoiding periodic angular bands.
    points[1]=(np.arange(count,dtype=np.float64)+points[1])/count
    return points


@contextmanager
def independent_launch_directions(seed):
    module=importlib.import_module('sionna.rt.radio_map_solvers.radio_map_solver')
    original=module.fibonacci_lattice
    def lattice(count):
        return mi.Point2f(stratified_square(count,seed))
    module.fibonacci_lattice=lattice
    try:
        yield
    finally:
        module.fibonacci_lattice=original
