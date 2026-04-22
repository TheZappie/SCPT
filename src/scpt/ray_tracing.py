from functools import partial

import numpy as np
from scipy.optimize import newton, brentq

from scpt.processing import strictly_increasing, all_positive


class RayTracingError(Exception):
    """Error related to impossible path between given source and receiver with given velocity model"""


def compute_horizontal_distance(p, h, s):
    """
    Compute horizontal distance for given ray parameter p.

    Args:
        p: ray parameter (s/m)
        h: array of layer thickness (m)
        s: array of layer slowness (s/m)

    Returns:
        X: horizontal distance (km)
    """
    if len(h) != len(s):
        raise ValueError(
            f"Length of thicknesses array should be same as slowness array {len(h)} != {len(s)}"
        )
    X = 0.0
    for i in range(len(h)):
        denominator = s[i] ** 2 - p**2

        if denominator <= 0:
            # Ray becomes evanescent in this layer
            raise RayTracingError(
                f"For the {i} layer, slowness < ray parameter. No ray path possible. "
            )

        X += p * h[i] / np.sqrt(denominator)

    return X


def compute_horizontal_distance_derivative(p, h, s):
    """
    Compute derivative dX/dp for given ray parameter p.

    Args:
        p: ray parameter (s/m)
        h: array of layer thickness (m)
        s: array of layer slowness (s/m)

    Returns:
        dX/dp: derivative of horizontal distance with respect to ray parameter p
    """
    if len(h) != len(s):
        raise ValueError(
            f"Length of thicknesses array should be same as slowness array {len(h)} != {len(s)}"
        )
    dXdp = 0.0
    for i in range(len(h)):
        denom = s[i] ** 2 - p**2

        if denom <= 0:
            raise RayTracingError(
                f"For the {i} layer, slowness < ray parameter. No ray path possible. "
            )

        dXdp += h[i] * s[i] ** 2 / (denom ** (3 / 2))

    return dXdp


def solve_ray_parameter(horizontal_distance, h, s, maxiter=50):
    """
    Solve for ray parameter p given target horizontal distance X.

    Args:
        horizontal_distance: target horizontal distance (km)
        h: array of layer thicknesses (km)
        s: array of layer slownesses (s/km)
        tol: convergence tolerance
        maxiter: maximum iterations

    Returns:
        p: ray parameter (s/km)
    """
    h = np.asarray(h)
    s = np.asarray(s)
    if len(h) != len(s):
        raise ValueError(
            f"Length of thicknesses array should be same as slowness array. Thickness: {len(h)}, slowness {len(s)}"
        )
    if np.any(s <= 0):
        raise RayTracingError("Slowness model should be positive")
    if np.any(h <= 0):
        raise RayTracingError("Model layer thickness should be positive")

    # Define the function to find root of: f(p) = X(p) - X_target
    def f(p):
        return compute_horizontal_distance(p, h, s) - horizontal_distance

    # Determine bounds
    s_min = np.min(s)  # Minimum slowness (maximum velocity)
    p_max = s_min - 1e-10

    try:
        if True:
            p_solution = brentq(f, 0, p_max, maxiter=maxiter)
        else:
            fprime = partial(compute_horizontal_distance_derivative, h=h, s=s)
            z_total = np.sum(h)
            # Straight-line approximation based on root-mean-square of the slowness
            p_init = (
                horizontal_distance
                / np.sqrt(horizontal_distance**2 + z_total**2)
                * np.sqrt(np.mean(s**2))
            )
            # Make sure it's within bounds
            p_init = min(p_init, p_max)
            p_solution = newton(f, p_init, fprime=fprime, maxiter=maxiter)

        # Check if solution is physical
        if p_solution < 0 or p_solution > p_max:
            raise ValueError(f"Solution out of physical bounds: p={p_solution}")

        return p_solution

    except RuntimeError as e:
        raise RayTracingError(f"Ray tracing algorithm failed to converge: {e}")


def solve_ray_parameter_between_points(x1, y1, x2, y2, h, s, maxiter: int = 50):
    """
    Solve for ray parameter p given two arbitrary points (x1,y1) and (x2,y2).
    ray parameter is related to the takeoff angle (angle of incidence) at the source, measured from the vertical.

    Args:
        x1, y1: starting point coordinates (m)
        x2, y2: ending point coordinates (m)
        h: array of layer boundaries (m)
        s: array of layer slownesses (s/m)
        maxiter: maximum iterations

    Returns:
        p: ray parameter (s/m)

    Details:
    1) Extracts the relevant depth range between y1 and y2
    2) Builds an "effective" velocity model by computing which portions of each layer fall within the depth range
    3) Calls original function with this adjusted model
    """
    h = np.array(h)
    s = np.array(s)

    if len(h) != len(s) + 1:
        raise ValueError("h must be one element longer than s.")
    if not all_positive(s):
        raise ValueError("Slowness model should be positive")
    if not strictly_increasing(h):
        raise ValueError("Model layer boundaries should be positive")
    if not (h[0] <= y1 <= h[-1]):
        raise ValueError(
            f"Model boundaries should encompass y1 (Given: {y1}). Current bounds: [{h[0]} - {h[-1]}]"
        )
    if not (h[0] <= y2 <= h[-1]):
        raise ValueError(
            f"Model boundaries should encompass y2 (Given: {y2}). Current bounds: [{h[0]} - {h[-1]}]"
        )

    if y1 == y2:
        # s of layer.
        return s[np.searchsorted(h, y1)]

    # Calculate horizontal distance and depth range
    horizontal_distance = abs(x2 - x1)
    y_start = min(y1, y2)
    y_end = max(y1, y2)

    layer_thickness = np.diff(h)

    # Find which layers are intersected by the depth range [y_start, y_end]
    # and create effective model for this depth range
    effective_h = []
    effective_s = []

    for i in range(len(layer_thickness)):
        layer_top = h[i]
        layer_bottom = h[i + 1]

        # Check if this layer intersects our depth range
        if layer_bottom <= y_start or layer_top >= y_end:
            continue  # Skip layers outside our range

        # Calculate intersection
        intersection_top = max(layer_top, y_start)
        intersection_bottom = min(layer_bottom, y_end)
        intersection_thickness = intersection_bottom - intersection_top

        if intersection_thickness > 0:
            effective_h.append(intersection_thickness)
            effective_s.append(s[i])

    if len(effective_h) == 0:
        raise RayTracingError(
            f"No valid layers found between depths {y_start} and {y_end}"
        )

    effective_h = np.array(effective_h)
    effective_s = np.array(effective_s)

    # Now solve with the effective model
    return solve_ray_parameter(
        horizontal_distance,
        effective_h,
        effective_s,
        maxiter=maxiter,
    )


def compute_ray_lengths_simple(p, thickness, s):
    """
    Compute the ray path length in each layer.

    The ray travels at an angle through each layer. This function computes
    the slant distance (path length) in each layer.

    Args:
        p: ray parameter (s/m)
        thickness: array of layer thickness (m)
        s: array of layer slowness (s/m)

    Returns:
        lengths: array of ray path lengths in each layer (km)

    Note:
        The slant distance in layer i is: h_i / cos(theta_i)
        where theta_i is the angle from vertical
    """
    if len(thickness) != len(s):
        raise ValueError(
            f"Length of thicknesses array should be same as slowness array {len(thickness)} != {len(s)}"
        )
    if not all_positive(s):
        raise ValueError("Slowness model should be positive")
    if not all_positive(thickness):
        raise ValueError("Model layer thickness should be positive")

    n = len(thickness)
    lengths = np.zeros(n)

    for i in range(n):
        # Check if ray propagates in this layer
        if p >= s[i]:
            raise RayTracingError(
                f"For the {i} layer, slowness < ray parameter. No ray path possible. "
            )

        # Angle from vertical using Snell's law: sin(theta) = p / s
        sin_theta = p / s[i]

        # Cosine of angle
        cos_theta = np.sqrt(1.0 - sin_theta**2)

        # Slant distance = vertical thickness / cos(theta)
        lengths[i] = thickness[i] / cos_theta

    return lengths


def compute_ray_lengths(y1, y2, h, s, p=None, tol=1e-12):
    """
    Compute the ray-path length within each layer for a ray connecting (x1,y1) to (x2,y2),
    given the ray parameter p in a horizontally layered medium.

    - Assumes straight ray segments in each layer (no turning within the interval).
    - Requires propagating conditions in traversed layers: |p| < s_i.
    """
    h = np.asarray(h, dtype=float)
    s = np.asarray(s, dtype=float)

    if not strictly_increasing(h):
        raise ValueError(f"Layer model should be strictly increasing: {h}")
    if not all_positive(s):
        raise RayTracingError("Slowness model should be positive")
    if h.ndim != 1 or s.ndim != 1 or h.size != s.size + 1:
        raise ValueError(
            f"`h` must have length one more than `s` (got len(h)={h.size}, len(s)={s.size})."
        )

    # Depth interval between the points
    y_start, y_end = (y1, y2) if y1 <= y2 else (y2, y1)

    # Layer tops/bottoms
    tops = h[:-1]
    bots = h[1:]

    # Intersection thickness per layer with [y_start, y_end]
    inter_top = np.maximum(tops, y_start)
    inter_bot = np.minimum(bots, y_end)
    dz = np.maximum(
        0.0, inter_bot - inter_top
    )  # vertical thickness traversed per layer

    if not np.any(dz > 0):
        raise ValueError(
            f"No valid layers intersect the depth interval [{y_start}, {y_end}]."
        )

    p_abs = abs(p)

    # Check propagating condition in traversed layers
    traversed = dz > 0
    if np.any(p_abs >= s[traversed] - tol):
        bad_layers = np.where(traversed & (p_abs >= s - tol))[0].tolist()
        raise ValueError(
            f"Ray parameter p={p} is not propagating in the traversed layer(s) {bad_layers}: requires |p| < s_i."
        )

    # Vertical slowness q_i and per-layer quantities
    q = np.sqrt(np.maximum(0.0, s**2 - p_abs**2))

    lengths = np.zeros_like(s)
    mask = dz > 0
    lengths[mask] = dz[mask] * s[mask] / q[mask]  # L_i

    return lengths


def compute_ray_path(y1: float, y2: float, layer_bounds, s, p):
    """
    Compute the (x,z) coordinates where the ray intersects each layer boundary
    between depths y1 and y2 in a horizontally layered medium.

    Parameters
    ----------
    y1, y2 : float
        Start and end depths (m).
    layer_bounds : array-like, shape (N+1,)
        Layer boundaries (m), increasing downward.
    s : array-like, shape (N,)
        Slowness in each layer (s/m).
    p : float
        Ray parameter (s/m).

    Returns
    -------
    x_coords : list of float
        Horizontal coordinate at each intersection point.
    z_coords : list of float
        Depths at each intersection point (layer boundaries).
    """

    layer_bounds = np.asarray(layer_bounds, float)
    s = np.asarray(s, float)

    if len(layer_bounds) != len(s) + 1:
        raise ValueError("h must be one element longer than s.")

    # Sort depths so the ray always propagates downward
    z_start, z_end = sorted([y1, y2])

    # Initial point
    x_coords = [0.0]
    z_coords = [z_start]

    current_x = 0.0
    current_z = z_start

    for i in range(len(s)):
        top = layer_bounds[i]
        bot = layer_bounds[i + 1]

        # Check if this layer is intersected by [z_start, z_end]
        if bot <= current_z or top >= z_end:
            continue

        # Intersection thickness inside this layer
        z_top = max(top, current_z)
        z_bot = min(bot, z_end)
        dz = z_bot - z_top

        if dz <= 0:
            continue

        # if (p >= s[i]) or np.isclose(s[i], p):
        #     raise ValueError(f"Ray becomes evanescent in layer {i + 1}")

        denom = np.sqrt(s[i] ** 2 - p**2)

        # Horizontal distance in this partial layer
        dx = dz * p / denom

        current_x += dx
        current_z = z_bot

        x_coords.append(current_x)
        z_coords.append(current_z)

        # Stop if we reached the end
        if current_z >= z_end - 1e-12:
            break

    return x_coords, z_coords


def compute_path_from_points(x1, y1, x2, y2, h, s, p=None):
    """
    Compute per-layer vertical (dz) and horizontal (dx) distances

    x1, y1: starting point coordinates (m)
    x2, y2: ending point coordinates (m)
    h: array of layer boundaries (m)
    s: array of layer slownesses (s/m)
    """

    h = np.asarray(h, dtype=float)
    s = np.asarray(s, dtype=float)

    if len(h) != len(s) + 1:
        raise ValueError("h must be one element longer than s.")
    if not all_positive(s):
        raise ValueError("Slowness model should be positive")
    if not strictly_increasing(h):
        raise ValueError("Model layer boundaries should be positive")
    if not (h[0] <= y1 <= h[-1]):
        raise ValueError(
            f"Model boundaries should encompass y1 (Given: {y1}). Current bounds: [{h[0]} - {h[-1]}]"
        )
    if not (h[0] <= y2 <= h[-1]):
        raise ValueError(
            f"Model boundaries should encompass y2 (Given: {y2}). Current bounds: [{h[0]} - {h[-1]}]"
        )

    if p is None:
        p = solve_ray_parameter_between_points(x1, y1, x2, y2, h, s)

    y_start, y_end = (y1, y2) if y1 <= y2 else (y2, y1)
    tops = h[:-1]
    bots = h[1:]

    dz = np.maximum(0.0, np.minimum(bots, y_end) - np.maximum(tops, y_start))

    if not np.any(dz > 0):
        return dz, np.zeros_like(dz)

    Δx_total = abs(x2 - x1)

    passed_layers = np.nonzero(dz)
    p_abs = abs(p)
    q = np.sqrt(np.maximum(0.0, s[passed_layers] ** 2 - p_abs**2))

    # --- Regular layers: ray theory ---
    dx = np.zeros_like(dz)
    dx[passed_layers] = dz[passed_layers] * p_abs / q

    grazing = dx > Δx_total

    # --- Grazing layers: constrained allocation ---
    dx_remaining = Δx_total - dx[~grazing].sum()

    if np.any(grazing):
        dz_grazing = dz[grazing].sum()
        if dz_grazing > 0:
            dx[grazing] = dx_remaining * dz[grazing] / dz_grazing
    return dz, dx


def compute_ray_lengths_safe(x1, y1, x2, y2, h, s, p=None):
    if np.hypot(x2 - x1, y2 - y1) == 0:
        return np.zeros_like(s)
    dz, dx = compute_path_from_points(x1, y1, x2, y2, h, s, p)
    return np.hypot(dx, dz)


def compute_travel_time(x1, y1, x2, y2, h, s, p=None):
    ray_segments = compute_ray_lengths_safe(x1, y1, x2, y2, h, s, p)
    return sum(ray_segments * s)


def compute_ray_path_safe(x1, y1, x2, y2, h, s, p=None):
    dz, dx = compute_path_from_points(x1, y1, x2, y2, h, s, p)
    if y1 == y2:
        return [[x1, x2], [y1, y2]]

    sign_x = np.sign(x2 - x1) or 1.0
    sign_z = np.sign(y2 - y1) or 1.0

    x_coords = [x1]
    z_coords = [y1]

    x, z = x1, y1

    for i in range(len(dz)):
        if dz[i] == 0:
            continue
        x += sign_x * dx[i]
        z += sign_z * dz[i]
        x_coords.append(x)
        z_coords.append(z)

    # enforce exact endpoint
    x_coords[-1] = x2
    z_coords[-1] = y2

    return np.asarray(x_coords), np.asarray(z_coords)
