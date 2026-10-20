"""Small CPU mirrors of JIT device properties used repeatedly by the GUI.

Retain a copy of the JIT value as well as its variable indices. This prevents
indices being recycled and makes in-place edits trigger Dr.Jit's copy-on-write.
Only immutable copies are returned; numerical solvers still use original arrays.
"""
import numpy as np


def _indices(value):
    index = getattr(value, 'index', 0)
    return (index,) if index else tuple(getattr(v, 'index', 0) for v in value)


def set_host_position(owner, position):
    """Assign a known CPU position without downloading it from the GPU again.

    Device positions are Point3f (float32). Seed only after assignment and
    make_opaque, using the same rounded data and the actual resulting JIT indices.
    Later assignments or component edits still invalidate the normal mirror.
    """
    import drjit as dr
    array = np.asarray(position, dtype=np.float32).reshape(3, 1).copy()
    owner.position = array.ravel()
    value = owner.position
    dr.make_opaque(value)
    indices = _indices(value)
    if indices and all(indices):
        cache = getattr(owner, '_gui_host_values', None)
        if cache is None:
            cache = {}
            owner._gui_host_values = cache
        array.setflags(write=False)
        cache['position'] = (indices, type(value)(value), array)


def host_value(owner, name):
    value = getattr(owner, name)
    indices = _indices(value)
    if not indices or not all(indices):
        return np.asarray(value.numpy()).copy()
    cache = getattr(owner, '_gui_host_values', None)
    if cache is None:
        cache = {}
        owner._gui_host_values = cache
    old = cache.get(name)
    if old is None or old[0] != indices:
        array = np.asarray(value.numpy()).copy()
        array.setflags(write=False)
        old = (indices, type(value)(value), array)
        cache[name] = old
    return old[2]
