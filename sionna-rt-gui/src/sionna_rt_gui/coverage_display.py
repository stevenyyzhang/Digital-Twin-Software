"""Display-only interpolation, with no invented values in unknown cells."""
import numpy as np

# Display-only sRGB. Unknown cells retain their original zero/invalid estimate.
UNKNOWN_RGB = (.16, .20, .36)


def vertex_gains(gain,mask):
    valid=mask & np.isfinite(gain) & (gain>0)
    rows,cols=gain.shape
    total=np.zeros((rows+1,cols+1));count=np.zeros_like(total)
    values=np.where(valid,gain,0)
    for y,x in [(0,0),(0,1),(1,1),(1,0)]:
        total[y:y+rows,x:x+cols]+=values
        count[y:y+rows,x:x+cols]+=valid
    mean=total/np.maximum(count,1)
    # Only blend corners fully surrounded by known outdoor cells. A missing cell,
    # roof or courtyard boundary prevents smoothing across that corner.
    corners=[np.where(count[y:y+rows,x:x+cols]==4,mean[y:y+rows,x:x+cols],gain)
             for y,x in [(0,0),(0,1),(1,1),(1,0)]]
    return np.stack(corners+[gain],axis=-1)[mask]


def sample_grid(gain, mask, x, y, smooth=False):
    """Sample in cell-edge coordinates; unknown/excluded cells stay unknown.

    Smoothing uses the same five-vertex cell fan as the 3D overlay. The
    returned values are display samples; the original grid is never changed.
    """
    rows, cols = gain.shape
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    inside = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    cx, cy = np.clip(ix, 0, cols-1), np.clip(iy, 0, rows-1)
    outdoor = inside & mask[cy, cx]
    values = gain[cy, cx].copy()
    valid = outdoor & np.isfinite(values) & (values > 0)
    if smooth:
        vertices = np.zeros((*gain.shape, 5))
        vertices[mask] = vertex_gains(gain, mask)
        v = vertices[cy, cx]
        u, t = x-ix-.5, y-iy-.5
        horizontal = np.abs(u) > np.abs(t)
        first = np.where(horizontal, np.where(u > 0, 1, 3), np.where(t > 0, 2, 0))
        second = (first+1) % 4
        edge = 2*np.maximum(np.abs(u), np.abs(t))
        a = np.where(horizontal, np.where(u > 0, u-t, -u+t), np.where(t > 0, t+u, -t-u))
        b = edge-a
        values = (np.take_along_axis(v, first[..., None], axis=-1)[..., 0]*a
                  + np.take_along_axis(v, second[..., None], axis=-1)[..., 0]*b
                  + v[..., 4]*(1-edge))
    return np.where(valid, values, np.nan), outdoor


def plan_overlay(gain, mask, origin, basis, low, side, resolution, power_dbm, cfg):
    """Project a planar RF grid onto an XY plan, preserving its outdoor mask."""
    import matplotlib
    axis = (np.arange(resolution)+.5)/resolution
    x, y = np.meshgrid(low[0]+axis*side, low[1]+(1-axis)*side)
    points = np.stack([x, y], axis=-1)
    cells = (points-origin) @ np.linalg.inv(basis).T
    values, outdoor = sample_grid(gain, mask, cells[..., 0], cells[..., 1], False)
    valid = np.isfinite(values) & (values > 0)
    rgba = np.zeros((resolution, resolution, 4), dtype=np.float32)
    # An explicit hatch separates excluded areas from unsampled ground cells.
    yy,xx=np.indices((resolution,resolution))
    hatch=(xx+yy)%18<3
    rgba[~outdoor]=[.06,.08,.10,.42]
    rgba[(~outdoor)&hatch]=[.12,.15,.18,.65]
    rgba[outdoor] = [*UNKNOWN_RGB, .88]
    level = (10*np.log10(values[valid])+power_dbm-cfg.vmin)/(cfg.vmax-cfg.vmin)
    rgba[valid, :3] = matplotlib.colormaps[cfg.color_map](np.clip(level, 0, 1))[:, :3]
    rgba[valid, 3] = .80
    return rgba
