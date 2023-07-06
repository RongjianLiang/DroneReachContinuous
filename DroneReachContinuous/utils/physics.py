import numpy as np

"""
Compute new speed, direction and location for spheres after impact and collision
"""

X_FLIP = np.array([-1, 1, 1])
Y_FLIP = np.array([1, -1, 1])
Z_FLIP = np.array([1, 1, -1])


def resolve_sphere_impact(
        sphere_old_speed: np.ndarray,
        coming_side: str,
        coef_restitution: float = 1.0):
    """
    Given sphere impact information, resolve the impact by calculating new speed, direction
    Args:
        sphere_old_speed: original sphere speed vector before collision
        coef_restitution: speed magnitude damping factor.
        coming_side: a string denoting what type of surface the sphere is impacting on.


    Returns:
        new sphere speed vector and new final location.

    """
    assert 0.0 < coef_restitution <= 1.0

    # Determine new speed value and direction
    if coming_side == "top" or coming_side == "bottom":
        sphere_new_speed = np.multiply(sphere_old_speed, coef_restitution * Z_FLIP)
    elif coming_side == "right" or coming_side == "left":
        sphere_new_speed = np.multiply(sphere_old_speed, coef_restitution * Y_FLIP)
    elif coming_side == "up" or coming_side == "down":
        sphere_new_speed = np.multiply(sphere_old_speed, coef_restitution * X_FLIP)
    else:
        raise ValueError("Invalid impact type")

    # Calculate new location
    # temp_new_loc = sphere_coords + t_value * sphere_old_speed + (1 - t_value) * sphere_new_speed
    return sphere_new_speed


def resolve_sphere_collision(
        sphere1_coords: np.ndarray,
        sphere1_old_speed: np.ndarray,
        sphere1_t_value: float,
        sphere2_coords: np.ndarray,
        sphere2_old_speed: np.ndarray,
        sphere2_t_value: float,
        coef_restitution: float = 1.0,
        sphere1_mass: float = 1.0,
        sphere2_mass: float = 1.0):
    """
    Calculate new spheres center coordinates and speeds vector after the collision.
    Args:
        sphere1_mass:
        sphere2_mass:
        coef_restitution:
        sphere1_coords: sphere 1 center coordinates at collision.
        sphere1_old_speed: sphere 1 old speed vector at collision.
        sphere1_t_value: coordinate t value.
        sphere2_coords: sphere 2 center coordinates at collision.
        sphere2_old_speed: sphere 2 old speed vector at collision.
        sphere2_t_value: coordinate t value


    Returns:
        New speed vectors and interpolated location after the collision.

    """
    # print(f"--t1: {sphere1_t_value}")
    # print(f"--t2: {sphere2_t_value}")
    assert 0 < coef_restitution <= 1.0
    assert 0 <= sphere1_t_value <= 1.0
    assert 0 <= sphere2_t_value <= 1.0
    assert sphere1_mass > 0
    assert sphere2_mass > 0

    # Calculate the relative position and speed vectors
    rel_pos = sphere2_coords - sphere1_coords
    rel_speed = sphere2_old_speed - sphere1_old_speed

    # Calculate the normalized collision normal vector
    collision_normal = rel_pos / np.linalg.norm(rel_pos)

    # Calculate the relative speed along the collision normal
    speed_along_normal = np.dot(rel_speed, collision_normal)

    # Calculate the impulse scalar
    impulse_scalar = (1 + coef_restitution) * speed_along_normal / \
                     (1 / sphere1_mass + 1 / sphere2_mass)

    # Calculate the impulse vector
    impulse = impulse_scalar * collision_normal

    # Calculate the new speed vectors
    new_speed1 = sphere1_old_speed + impulse / sphere1_mass
    new_speed2 = sphere2_old_speed - impulse / sphere2_mass

    # Calculate the new temp location
    # new_loc1 = sphere1_coords + (1 - sphere1_t_value) * new_speed1
    # new_loc2 = sphere2_coords + (1 - sphere2_t_value) * new_speed2

    return new_speed1, new_speed2




