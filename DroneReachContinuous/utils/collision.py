import numpy as np

from geometry import *
from physics import *

array = np.array
"""
This new collision detection algorithms are based on finding intersection point between 
line segments and bounded surfaces. 
It uses a broad-phase check that first detect collision
between axis-aligned bounding box, then for potential collision uses continuous collision detection 
to get exact solution. The collision response is calculated using inelastic collision models.
"""

"""Declare useful normal vectors"""
Y_Z_NORM_OUT = array([1, 0, 0])
Y_Z_NORM_IN = array([-1, 0, 0])
X_Z_NORM_OUT = array([0, 1, 0])
X_Z_NORM_IN = array([0, -1, 0])
X_Y_NORM_OUT = array([0, 0, 1])
X_Y_NORM_IN = array([0, 0, -1])

IMPACT_EVENT = 1
COLLISION_EVENT = 2
MAX_ITER = 5


def find_sphere_sphere_collision(
        start1: np.ndarray,
        end1: np.ndarray,
        start2: np.ndarray,
        end2: np.ndarray,
        speed1: np.ndarray,
        speed2: np.ndarray,
        radius1: float,
        radius2: float):
    """
    Given information about spheres' traversal and their closest positions to each
    other, check if these positions meet collision requirement.
    If met, rewind their traversal from the closest positions to solve the exact positions of collision.
    Else, return None.
    Args:
        start1: start coordinate of sphere 1.
        end1: end coordinate of sphere 1.
        start2: start coordinate of sphere 2.
        end2: end coordinate of sphere 2.
        speed1: speed vector of sphere 1.
        speed2: speed vector of sphere 2.
        radius1: radius of sphere 1.
        radius2: radius of sphere 2.

    Returns:
        two info_dict, where
        info_dict_1 = {
        "t": t, "pos": collision_position_1} and
        info_dict_2 = {
        "t": u, "pos": collision_position_2}

    """
    closest_result = find_closest_positions(start1, end1, start2, end2)

    if closest_result is None:
        # collinear or parallel case
        if np.linalg.norm(start1 - start2) <= radius1 + radius2:
            # collide at start
            # todo: return the dict!
            info_dict_1 = {
                "t": 0, "pos": start1
            }
            info_dict_2 = {
                "t": 0, "pos": start2
            }
            return info_dict_1, info_dict_2
        else:
            # collide for collinear

            start_distance = np.linalg.norm(start1 - start2)
            clearance = radius2 + radius1
            approach_distance = start_distance - clearance
            approach_speed = np.linalg.norm(speed1 - speed2)
            approach_time = approach_distance / approach_speed

            # print("-----checking: collinear case")
            # print(f"------start dis: {start_distance}")
            # print(f"------approach dis: {approach_distance}")
            # print(f"------approach speed: {approach_speed}")
            # print(f"------approach time: {approach_time}")

    else:
        # general case
        closest_position1, closest_position2 = closest_result

        closest_vector = closest_position1 - closest_position2
        closest_distance = np.linalg.norm(closest_vector)

        # check if a collision is possible
        if not closest_distance <= radius1 + radius2:
            return None

        # solve for exact collision time using relative distance and speed
        start_vector = start1 - start2
        start_distance = np.linalg.norm(start_vector)
        clearance = radius2 + radius1

        approach_vector1 = closest_vector / closest_distance
        approach_vector2 = - approach_vector1

        approach_speed1 = np.dot(speed1, approach_vector1)
        approach_speed2 = np.dot(speed2, approach_vector2)

        approach_time = (start_distance - clearance) / (approach_speed1 + approach_speed2)

    covered_1 = approach_time * speed1
    covered_2 = approach_time * speed2
    seg1 = end1 - start1
    seg2 = end2 - start2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sum(np.nan_to_num(covered_1 / seg1, nan=0.0, neginf=0.0, posinf=0.0))
        u = np.sum(np.nan_to_num(covered_2 / seg2, nan=0.0, neginf=0.0, posinf=0.0))

    collision_position_1 = start1 + covered_1
    collision_position_2 = start2 + covered_2

    info_dict_1 = {
        "t": t, "pos": collision_position_1
    }
    info_dict_2 = {
        "t": u, "pos": collision_position_2
    }
    return info_dict_1, info_dict_2


def helper_find_sphere_edge_collision_side(
        sphere_radius: float,
        line_start: np.ndarray,
        line_norm: float,
        line_unit_dir: np.ndarray,
        edge_info: list):
    """
    Given the sphere traversal information and potential set of
    edges that would collide, find the sphere center coordinate
    when collides.
    *Assume no "center" collision on edge. Assume and solve for
    "side" collision with edge.*
    Args:
        sphere_radius: sphere radius
        line_start: start coordinate of the line segment.
        line_norm:
        line_unit_dir:
        edge_info: a list of edge information.

    Returns:
        collision point and the t value of sphere center when collides.
    """

    edge_seg_distance = edge_info[1]
    seg_closest_pos = edge_info[2]
    delta_seg = np.sqrt(
        sphere_radius ** 2 - edge_seg_distance ** 2
    )
    delta_seg_vector = array(
        - np.dot(line_unit_dir, delta_seg)
    )
    sphere_coords = seg_closest_pos + delta_seg_vector
    t = (np.linalg.norm(line_start - sphere_coords) / line_norm)

    return sphere_coords, t


def helper_find_sphere_edge_collision_center(
        sphere_radius: float,
        line_start: np.ndarray,
        line_norm: float,
        line_unit_dir: array,
        intersect_res: np.ndarray):
    """
    Given the sphere traversal information and potential set of
    edges that would collide, find the sphere center coordinate
    when collides.
    *Assume no "side" collision on edge. Assume and solve for
    "center" collision with edge.*
    Args:
        sphere_radius:
        line_start:
        line_norm:
        line_unit_dir:
        intersect_res:

    Returns:

    """
    # intersection point is the impact point
    # just rewind from intersection point to get sphere coords
    # print(f"intersect_res: {intersect_res}")
    sphere_center_at_impact = array(intersect_res - line_unit_dir * sphere_radius)
    t = np.linalg.norm(sphere_center_at_impact - line_start) / line_norm
    return sphere_center_at_impact, t


def helper_sphere_surface_validity_check(
        sphere_center_at_impact: np.ndarray,
        t: float,
        coming_side: str,
        surface_boundary: dict):
    # print(f"------===--- coming side: {coming_side}\n"
    #       f"------===--- surface boundary: \n{surface_boundary}\n")
    x_min = surface_boundary["x"][0]
    x_max = surface_boundary["x"][1]
    y_min = surface_boundary["y"][0]
    y_max = surface_boundary["y"][1]
    z_min = surface_boundary["z"][0]
    z_max = surface_boundary["z"][1]

    valid_t = np.logical_and(0 <= t, t <= 1)
    # print("intersection:", sphere_center_at_impact)
    # Check if the intersection point is within the surface's bounds
    if coming_side == "left" or coming_side == "right":
        valid_bounds = np.logical_and.reduce((
            x_min <= sphere_center_at_impact[0],
            x_max >= sphere_center_at_impact[0],
            z_min <= sphere_center_at_impact[2],
            z_max >= sphere_center_at_impact[2])
        )

    elif coming_side == "down" or coming_side == "up":
        valid_bounds = np.logical_and.reduce((
            y_min <= sphere_center_at_impact[1],
            y_max >= sphere_center_at_impact[1],
            z_min <= sphere_center_at_impact[2],
            z_max >= sphere_center_at_impact[2])
        )
    elif coming_side == "top" or coming_side == "bottom":
        valid_bounds = np.logical_and.reduce((
            x_min <= sphere_center_at_impact[0],
            x_max >= sphere_center_at_impact[0],
            y_min <= sphere_center_at_impact[1],
            y_max >= sphere_center_at_impact[1])
        )
    else:
        raise ValueError("Invalid surface type")

    return valid_t & valid_bounds


# todo: currently this function return the impact coords instead
def helper_check_sphere_edge_collision(
        line_start: np.ndarray,
        line_end: np.ndarray,
        sphere_radius: float,
        edges: list):
    """
    Check if the edges are involved in the collision, either a center
    collision where sphere trajectory intersect with edges or a side
    collision where sphere touches the edges. In the case of multiple
    collisions, only the earliest collision is valid.
    Args:
        line_start: sphere trajectory start coordinate.
        line_end:  sphere trajectory end coordinate.
        sphere_radius: sphere radius.
        edges: a list of surface edges.

    Returns:

    """
    center_collision_res = {}
    side_collision_res = {}

    # compute line info
    line_norm = (np.linalg.norm(line_end - line_start))
    line_unit_dir = ((line_end - line_start) / line_norm)

    for edge in edges:
        # first check if the trajectory intersects with edge
        intersect_res = helper_find_line_edge_intersection(line_start, line_end, line_unit_dir, edge[0], edge[1],
                                                           edge[1] - edge[0])

        if intersect_res is not None:
            # print(f"---checking: {edge} intersection result: {intersect_res}")
            center_res, t = helper_find_sphere_edge_collision_center(
                sphere_radius, line_start, line_norm, line_unit_dir, intersect_res)
            center_collision_res[t] = [center_res, t]
        else:
            # todo: the following has problem as well...
            closest_res = find_closest_positions(line_start, line_end, edge[0], edge[1])
            if closest_res is not None:

                pos1, pos2 = closest_res
                edge_seg_distance = np.linalg.norm((pos1 - pos2))
                if edge_seg_distance <= sphere_radius:
                    # run into edge
                    edge_info = [edge, edge_seg_distance, pos1, pos2]
                    side_res, t = helper_find_sphere_edge_collision_side(
                        sphere_radius, line_start, line_norm, line_unit_dir, edge_info)
                    # print(f"\n---checking: {edge} closest: {closest_res}, dis: {edge_seg_distance}, "
                    #       f"side res: {t, side_res}\n")
                    side_collision_res[t] = [side_res, t]

    # Merge the result, sort by t-value and get the earliest
    edge_collision_res = {**center_collision_res, **side_collision_res}
    if len(edge_collision_res) == 0:
        return None
    else:
        result = list(sorted(edge_collision_res.items()))
        sphere_coords, t = result[0][1]
        # print(f"++edge result: {t, sphere_coords}")
        return sphere_coords, t


def find_sphere_surface_impact(
        sphere_radius: float,
        line_start: np.ndarray,
        line_end: np.ndarray,
        surface_centroid: np.ndarray,
        surface_boundary_dict: dict,
        surface_edges: list,
        coming_side: str):
    """
    An optimized way to find the sphere center coordinates when it traverses on given line segment
    and contact the surface, exploiting the fact that all surfaces in questions
    are parallel to one of the x-y, x-z, y-z planes.
    Args:
        coming_side:
        surface_edges:
        surface_boundary_dict:
        sphere_radius: radius of the sphere.
        line_start: start coordinates of line segment.
        line_end: end coordinate of line segment.
        surface_centroid: surface centroid.

    Returns:
        The sphere center coordinate when contact the surface, and its t-value on the line segment.
    """
    # determine if and how the edges are involved in the collision
    edge_collision_result = helper_check_sphere_edge_collision(
        line_start, line_end, sphere_radius, surface_edges
    )
    # todo: use coming side instead!
    # Need to incorporate t-checking between edge and surface collision
    # print(f"edge collision check result: {edge_collision_result}")
    # if edge_collision_result is not None:
    #     return edge_collision_result

    # General sphere-surface collision
    # Extract the constant coordinate value based on the surface type
    # Move the surface closer to the trajectory along normal direction, by a distance of sphere radius

    # use dictionary instead?
    # print(f"===   ===   ===   === given coming side: {coming_side}")
    if coming_side == "down":
        constant_coord = surface_centroid[0] + sphere_radius
    elif coming_side == "up":
        constant_coord = surface_centroid[0] - sphere_radius
    elif coming_side == "left":
        constant_coord = surface_centroid[1] - sphere_radius
    elif coming_side == "right":
        constant_coord = surface_centroid[1] + sphere_radius
    elif coming_side == "top":
        constant_coord = surface_centroid[2] + sphere_radius
    elif coming_side == "bottom":
        constant_coord = surface_centroid[2] - sphere_radius
    else:
        raise ValueError("Invalid coming side")
    # Compute the intersection point based on the surface type and constant coordinate

    with np.errstate(divide='ignore', invalid='ignore'):
        if coming_side == "down" or coming_side == "up":
            t = (constant_coord - line_start[0]) / (line_end[0] - line_start[0])
            sphere_center_at_impact = line_start + t * (line_end - line_start)
            # sphere_center_at_impact[0] = constant_coord
            # print(f"===   ===   ===   === checking up and down: {sphere_center_at_impact}")
        elif coming_side == "left" or coming_side == "right":
            t = (constant_coord - line_start[1]) / (line_end[1] - line_start[1])
            sphere_center_at_impact = line_start + t * (line_end - line_start)
            # sphere_center_at_impact[1] = constant_coord
            # print(f"\n=========   === checking left and right: {sphere_center_at_impact}\n"
            #       f"=========   === start: {line_start}\n"
            #       f"=========   === end: {line_end}\n"
            #       f"=========   === const: {constant_coord}\n"
            #       f"=========   === t-value: {t}\n")
        elif coming_side == "top" or coming_side == "bottom":
            t = (constant_coord - line_start[2]) / (line_end[2] - line_start[2])
            sphere_center_at_impact = line_start + t * (line_end - line_start)
            # sphere_center_at_impact[2] = constant_coord
            # print(f"===   ===   ===   === checking top and bottom: {sphere_center_at_impact}")
        else:
            raise ValueError("Invalid surface type")

    # Validity check for above results
    # print(f"===   ===   ===   === validity checking: {sphere_center_at_impact}")
    validity = helper_sphere_surface_validity_check(
        sphere_center_at_impact, t, coming_side, surface_boundary_dict
    )

    if edge_collision_result is not None:
        _, t_edge = edge_collision_result
        if validity:
            if t_edge > t:
                return sphere_center_at_impact, t
        else:
            # print(f"========= ===collision 378: not valid: "
            #       f"{sphere_center_at_impact, t, coming_side}")
            return edge_collision_result
    elif validity:
        return sphere_center_at_impact, t
    else:
        # print(f"========= ===collision 378: not valid: "
        #       f"{sphere_center_at_impact, t, coming_side}")
        return None


# todo: update this function the return type of surfaces the sphere impacts on
# todo: update all find impact/collision function to return info as dict
def sphere_building_impact(
        sphere_radius: float,
        sphere_loc1: np.ndarray,
        sphere_loc2: np.ndarray,
        building_centroid,
        building_info: dict):
    """
    calculate impact information between spheres and buildings if
    an impact exists, else return None.
    Args:
        building_centroid:
        sphere_radius:
        sphere_loc1:
        sphere_loc2:
        building_info:

    Returns:
        [t, info_dict] where
        info_dict = {
            "coords": coords, "t": t, "coming_side": coming_side
        }

    """
    # line_norm = np.linalg.norm(sphere_loc1 - sphere_loc2)
    # line_unit_dir = sphere_loc1 - sphere_loc2 / line_norm
    building_centroid = array(building_centroid)

    # Naive checking should be handled by broad-phase check separately
    coming_side = []
    impact_result = {}

    coming_dir = building_centroid - sphere_loc1
    if np.dot(coming_dir, X_Y_NORM_OUT) > 0:
        coming_side.append("bottom")
    if np.dot(coming_dir, X_Y_NORM_IN) > 0:
        coming_side.append("top")
    if np.dot(coming_dir, Y_Z_NORM_OUT) > 0:
        coming_side.append("up")
    if np.dot(coming_dir, Y_Z_NORM_IN) > 0:
        coming_side.append("down")
    if np.dot(coming_dir, X_Z_NORM_IN) > 0:
        coming_side.append("right")
    if np.dot(coming_dir, X_Z_NORM_OUT) > 0:
        coming_side.append("left")

    # print(f"=============== coming side: {coming_side}")
    # print(f"--------building info: {building_info}")
    for possible_side in coming_side:
        # extract information
        # if statement to avoid errors when checking borders
        if possible_side in building_info.keys():
            surface_centroid = building_info[possible_side]["surface_centroid"]
            surface_boundary = building_info[possible_side]["surface_boundary"]
            surface_edges = building_info[possible_side]["surface_edges"]
            # coming_side = building_info[possible_side]["coming_side"]
            # print(f"=============== checking coming side: {possible_side}")
            # print(f"=============== surface centroid: {surface_centroid}")
            surface_result = find_sphere_surface_impact(
                sphere_radius, sphere_loc1, sphere_loc2,
                surface_centroid,
                surface_boundary, surface_edges, possible_side)
            # print(f"=============== checking coming side: {possible_side}, result: {surface_result}")
            if surface_result is not None:
                sphere_coords, t = surface_result
                impact_result[t] = [sphere_coords, t, possible_side]

    if len(impact_result) > 0:

        temp = list(sorted(impact_result.items()))
        valid_temp_t = temp[0][0]
        valid_temp_info = temp[0][1]
        impact_result[valid_temp_t] = valid_temp_info
        coords, t, surface_type = valid_temp_info
        info_dict = {
            "coords": coords, "t": t, "coming_side": surface_type
        }
        # print(f"=============== validating: {building_centroid}: {info_dict}")
        return [valid_temp_t, info_dict]
    else:
        # print(f"=============== all possible sizes: {coming_side}, but only have {building_centroid}: {building_info.keys()}")
        return None


def broad_phase_check(
        num_spheres: int,
        s_aabb_centroids,
        s_aabb_dim,
        buildings_info,
        buildings_aabb):
    impact_results = []
    collision_results = []
    for s_id in range(num_spheres):
        for buildings in buildings_info.keys():
            building_aabb = buildings_aabb[buildings]
            sphere_aabb = [s_aabb_centroids[s_id],
                           s_aabb_dim[s_id]]

            if check_aabb_intersection(building_aabb, sphere_aabb):
                impact_results.append([s_id, buildings])

    for s_id1 in range(num_spheres):
        for s_id2 in range(s_id1 + 1, num_spheres):
            aabb1 = [s_aabb_centroids[s_id1],
                     s_aabb_dim[s_id1]]
            aabb2 = [s_aabb_centroids[s_id2],
                     s_aabb_dim[s_id2]]
            if check_aabb_intersection(aabb1, aabb2):
                collision_results.append([s_id1, s_id2])

    return impact_results, collision_results


def step_spheres(
        city_height: int,
        num_spheres: int,
        spheres_location1: list or np.ndarray,
        spheres_speeds: list or np.ndarray,
        buildings_info: dict,
        buildings_aabb: dict,
        spheres_radius: float,
        fps: int = 24,
        impact_coef_restitution: float = 1.0,
        collis_coef_restitution: float = 1.0,
):
    """
    Given the start and end locations of all spheres, spheres original speed vectors during
    the time range, as well as the city information dictionary, find and resolve all possible
    collision and impact within the time step.
    Args:
        fps:
        city_height:
        num_spheres:
        buildings_aabb: the aligned-axis-bounding-box representation of the city.
        spheres_radius: radius for all spheres. Assume uniform radius
        collis_coef_restitution: coefficient of restitution for collision.
        impact_coef_restitution: coefficient of restitution for impact.
        spheres_location1: sphere start location in a list.
        spheres_speeds:  sphere speed vectors.
        buildings_info:  a dictionary with city information. Keys should be the building's centroid,
        and values being the detailed information of the buildings, including its surfaces, edges,
        and axis-aligned-bounding-box representation.

    Returns:
        a list of spheres locations and speeds at the end of this step, as well as the recorders that
        stores all intermediate locations, speed vectors, and impact/collision dictionary.

    """
    # make a copy of init info

    # cast location and speeds to np arrays
    if isinstance(spheres_speeds, list):
        init_speed = array(spheres_speeds)
    else:
        init_speed = spheres_speeds
    if isinstance(spheres_location1, list):
        init_loc = array(spheres_location1)
    else:
        init_loc = spheres_location1

    intervals_per_step = fps - 1
    delta_t = 1.0 / intervals_per_step

    # init variables for the check, and init the recorder for this step
    frame_start_loc = init_loc
    _start_speed = init_speed
    loc_recorder = [frame_start_loc]
    speed_recorder = [_start_speed]
    event_recorder = [np.zeros(num_spheres)]

    # start stepping discretely
    for interval_id in range(intervals_per_step):

        # todo: break the loop earlier for testing
        # if interval_id == 1:
        #     break

        # todo: init the event recorder
        _sub_event = np.zeros(num_spheres)

        frame_end_loc = frame_start_loc + _start_speed * delta_t

        s_aabb_centroids, s_aabb_dim = \
            find_aabb_for_sphere(
                frame_start_loc, frame_end_loc, spheres_radius
            )
        # find intersection between an array of bounding boxes
        # note that contact between bounding boxes does not count as intersection
        i_broad_check, c_broad_check = broad_phase_check(
            num_spheres, s_aabb_centroids, s_aabb_dim, buildings_info, buildings_aabb
        )

        num_iter = 0
        temp = {}
        # get a copy of end location and init the try end loc for checking
        try_start_loc = frame_start_loc
        try_end_loc = frame_end_loc
        try_speed = _start_speed

        # todo: repeated impact detectors and override flag set here
        reversed_new_speed_in_loop = {_id: 0 for _id in range(num_spheres)}
        # todo: init the override flag
        override_new_speed_in_loop = {_id: False for _id in range(num_spheres)}
        # print(f"\n-- step: {interval_id}, i-check: {i_broad_check}, c-check: {c_broad_check}")
        # print(f"++ step: {interval_id}, current loc: \n{frame_start_loc}, \n end loc: \n{frame_end_loc}")
        # print(f"-- s_aabb: {s_aabb_centroids}, {s_aabb_dim}")

        while (len(i_broad_check) > 0 or len(c_broad_check) > 0) and num_iter < MAX_ITER:
            # todo: in the loop, need to further detect repeated collisions and impacts
            # todo: for repeated impacts, happen when the dimensions between surfaces are too small for the sphere
            # todo: scenario including placing a large sphere between surfaces or sphere at the corner between surfaces
            # todo: the second scenario can resolve itself by looping the detection-resolution
            # todo: but the first one needs explicit detection
            # todo: although this could be avoided by having small sphere diameter
            # todo: but it is still suggested to implement for more robust simulation.
            # todo: patterns for this would be shifting opposite speed directions (not magnitude!)
            # todo: so we would use intermediate resolved_speed for each agents
            # todo: and calculate if their new speeds (after resolved) are opposite to their previous ones
            # todo: and count how many times these are opposite
            # todo: once the numbers goes above 2, we know it is vibrating between surfaces
            # todo: the resolution would then be putting the the speed component on this conflicting direction to 0.
            # todo: there would be an override flag so that above results would override the resolver output
            # todo: for repeated collision, need more tests to understand its problem and implications to our solutions.

            # print(f"\n==== iter: {num_iter}, current: i: {i_broad_check}, c: {c_broad_check}")
            # print(f"==== iter: {num_iter}, current loc: \n{try_start_loc}, end loc: {try_end_loc}")
            # Use broad-phase results, conduct detailed check only on affected when necessary
            i_check_result = {}
            if len(i_broad_check) > 0:
                # impact detailed check
                # print(f"===+++iter: {num_iter} detailed impact check: {i_broad_check}")
                for possible_impact in i_broad_check:
                    s_id, building = possible_impact
                    s_loc1 = try_start_loc[s_id]
                    s_loc2 = try_end_loc[s_id]
                    if 0 < building[2] < city_height:       # usual buildings and vertical borders
                        b_axis = np.array([building[0], building[1], s_loc1[2]])
                    else:                                   # special top and bottom borders
                        b_axis = np.array([s_loc1[0], s_loc1[1], building[2]])
                    s_to_b = b_axis - s_loc1
                    # using building centroid to check for approaching is problematic, as the centroid does not
                    # represent extreme ends of the building very well.
                    if np.dot(s_to_b, s_loc2 - s_loc1) > 0:     # only check if approaching the building.
                        if isinstance(spheres_radius, list) or isinstance(spheres_radius, dict):
                            r = spheres_radius[s_id]
                        else:
                            r = spheres_radius
                        building_centroid = building
                        building_info = buildings_info[building]

                        impact_result = sphere_building_impact(
                            r, s_loc1, s_loc2, building_centroid,
                            building_info
                        )
                        # print(f"===iter: {num_iter}: checking building: {building}, result: {impact_result}")
                        if impact_result is not None:
                            t, info_dict = impact_result
                            info_dict["type"] = "impact"
                            i_check_result[s_id] = info_dict
                            # print(f"---+++step {interval_id}, iter: {num_iter} sphere coming at building {building}!\n"
                            #       f"---+++coming loc: from {s_loc1} to {s_loc2}\n"
                            #       f"---+++impact check: {impact_result}\n"
                            #       f"---+++breaking from check loop")
                            break

            # print(f"---===current i: {i_check_result}")
            c_check_result = {}
            c_pair = []
            if len(c_broad_check) > 0:
                # collision detailed check
                for possible_collision in c_broad_check:
                    s_id1, s_id2 = possible_collision

                    start1 = try_start_loc[s_id1]
                    start2 = try_start_loc[s_id2]
                    end1 = try_end_loc[s_id1]
                    end2 = try_end_loc[s_id2]
                    speed1 = try_speed[s_id1]
                    speed2 = try_speed[s_id2]
                    # print(f"====iter: {num_iter}, checking following for collision:")
                    # print(f"====start loc:====\n{array(start1), array(start2)}")
                    # print(f"====end loc:====\n{array(end1), array(end2)}\n")
                    if isinstance(spheres_radius, list) or isinstance(spheres_radius, dict):
                        r1 = spheres_radius[s_id1]
                        r2 = spheres_radius[s_id2]
                    else:
                        r1 = r2 = spheres_radius
                    # print(f"---c_check: \n{end1}\n{end2}")
                    collision_result = find_sphere_sphere_collision(
                        start1, end1,
                        start2, end2,
                        speed1, speed2,
                        r1, r2
                    )

                    if collision_result is not None:
                        res1, res2 = collision_result
                        # we also need to know who are we colliding with
                        # now the res is [t, coords, other_id]
                        # print(f"-----check result: {collision_result}")
                        # print(f"-----collision: \n{res1}\n{res2}\n")
                        c_pair.append({s_id1, s_id2})
                        # add new keys into the res dicts
                        res1["other_id"] = s_id2
                        res1["type"] = "collision"
                        res2["other_id"] = s_id1
                        res2["type"] = "collision"
                        # todo: need to check if s_id1 or s_id2 already in the result dict
                        if s_id1 in c_check_result:
                            t_ = c_check_result[s_id1]["t"]
                            if res1["t"] < t_:
                                c_check_result[s_id1] = res1
                            # if res1["t"] == t_:
                        if s_id2 in c_check_result:
                            t_ = c_check_result[s_id2]["t"]
                            if res2["t"] < t_:
                                c_check_result[s_id2] = res2
                        #
                        # the result dict is wrapped in a list
                        c_check_result[s_id1] = res1
                        c_check_result[s_id2] = res2
                        # todo: need to deal with multi-object collisions
                        # collision_check_results[(s_id1, s_id2)] = [res1, res2]

            # print(f"=== step: {interval_id} "
            #       f"iter: {num_iter}, c_result: {c_check_result}, i_result: {i_check_result}")

            # Given the s_id information, lets merge the dict based on t-value
            # Only the earlier info will be kept
            # In the case of same time impact-collision, collision supersedes impact
            # as the shared keys might not be too many, use this method
            # todo: update the event here
            temp = {**i_check_result, **c_check_result}

            if len(temp) > 0:
                _sub_event[list(temp.keys())] = [
                    COLLISION_EVENT if temp[_id]["type"] == "collision"
                    else IMPACT_EVENT for _id in temp.keys()
                ]
            else:
                pass

            if len(i_check_result) > 0 and len(c_check_result) > 0:
                # temp = {**i_check_result, **c_check_result}
                for s_id, info in temp.items():
                    if s_id in i_check_result:
                        impact_info = i_check_result[s_id]
                        if impact_info["t"] < info["t"]:
                            temp[s_id] = impact_info
            elif len(i_broad_check) > 0 and len(c_check_result) == 0:
                temp = i_check_result
            elif len(c_check_result) > 0 and len(i_check_result) == 0:
                temp = c_check_result
            else:
                # skip the following code in this loop
                # and return the check loop conditions
                continue
            # print(f"------temp: {temp}")
            # Resolving and updating the speed
            for s_id, info in temp.items():
                if info["type"] == "impact":
                    old_speed = try_speed[s_id]
                    s_coords = try_start_loc[s_id]
                    t_value = info["t"]
                    impact_type = info["coming_side"]
                    new_speed = resolve_sphere_impact(
                        old_speed, impact_type, impact_coef_restitution
                    )

                    # todo: check if new speed is opposite to the old speed, and update the counter for this agent
                    if np.dot(new_speed, old_speed) == - np.linalg.norm(new_speed) * np.linalg.norm(old_speed):
                        reversed_new_speed_in_loop[s_id] += 1
                        # print(f"------reversed new speed: {s_id}, No.: {reversed_new_speed_in_loop[s_id]}")
                    if reversed_new_speed_in_loop[s_id] >= 2:
                        override_new_speed_in_loop[s_id] = True
                    if override_new_speed_in_loop[s_id]:
                        # override this new speed
                        # print(f"------override: {s_id} new speed to 0")
                        new_speed = np.array([0, 0, 0])
                    try_speed[s_id] = new_speed
                    # incrementally update here! Do not use the new loc from resolve!!!
                    # update from the intermediate location
                    pre_impact_loc = s_coords + old_speed * delta_t * t_value
                    try_end_loc[s_id] = pre_impact_loc + new_speed * delta_t * (1 - t_value)
                    # print(f"--------resolved end loc: {pre_impact_loc + new_speed * delta_t * (1 - t_value)}")
                elif info["type"] == "collision":
                    s1_id = s_id
                    s2_id = info["other_id"]
                    if {s1_id, s2_id} in c_pair:
                        s1_speed = try_speed[s1_id]
                        s1_coords = info["pos"]    # use sphere coord at collision
                        s1_t_value = info["t"]
                        s2_info = temp[s2_id]
                        s2_speed = try_speed[s2_id]
                        s2_coords = s2_info["pos"]      # use sphere coord at collision
                        s2_t_value = s2_info["t"]
                        # assume same mass here of 1.0
                        new_speed1, new_speed2 = \
                            resolve_sphere_collision(
                                s1_coords, s1_speed, s1_t_value,
                                s2_coords, s2_speed, s2_t_value,
                                collis_coef_restitution
                            )

                        # check for repeated collision and reversed speed
                        if np.dot(s1_speed, new_speed1) == - np.linalg.norm(s1_speed) * np.linalg.norm(new_speed1):
                            reversed_new_speed_in_loop[s1_id] += 1
                        if np.dot(s2_speed, new_speed2) == - np.linalg.norm(s2_speed) * np.linalg.norm(new_speed2):
                            reversed_new_speed_in_loop[s2_id] += 1

                        if reversed_new_speed_in_loop[s1_id] >= 2:
                            override_new_speed_in_loop[s1_id] = True
                        if reversed_new_speed_in_loop[s2_id] >= 2:
                            override_new_speed_in_loop[s2_id] = True

                        if override_new_speed_in_loop[s1_id]:
                            new_speed1 = np.array([0, 0, 0])
                        if override_new_speed_in_loop[s2_id]:
                            new_speed2 = np.array([0, 0, 0])
                        # here we should directly update the _start_speed
                        try_speed[s1_id] = new_speed1
                        try_speed[s2_id] = new_speed2
                        # incrementally update here! Do not use the new loc from resolve!!!
                        # actually, we need to step them from the intermediate position
                        pre_collision_loc1 = s1_coords + s1_speed * delta_t * s1_t_value
                        pre_collision_loc2 = s2_coords + s2_speed * delta_t * s2_t_value
                        try_end_loc[s1_id] = pre_collision_loc1 + new_speed1 * delta_t * (1 - s1_t_value)
                        try_end_loc[s2_id] = pre_collision_loc2 + new_speed2 * delta_t * (1 - s2_t_value)
                        # this pair has been resolved, remove it from the pair list
                        c_pair.remove({s1_id, s2_id})
                    else:
                        # already being resolved
                        continue

            # check again, use the updated start speed
            _try_end_loc = frame_start_loc + _start_speed * delta_t
            # print(f"------try: start loc: {try_start_loc},   end loc: {try_end_loc}")
            # print(f"------iter broad phase check")
            _s_aabb_centroids, _s_aabb_dim = find_aabb_for_sphere(
                    try_start_loc, try_end_loc, spheres_radius
            )
            i_broad_check, c_broad_check = broad_phase_check(
                num_spheres, _s_aabb_centroids, _s_aabb_dim, buildings_info, buildings_aabb
            )
            # print(f"=====iter: {num_iter}, i-check: {i_broad_check}, c-check: {c_broad_check}\n")
            # update for next iteration of check
            num_iter += 1

        # update for next step
        frame_start_loc = try_end_loc
        _start_speed = try_speed
        # Record the step
        loc_recorder.append(try_end_loc)
        speed_recorder.append(try_speed)
        # todo: our event recorder should only record at the start of the step, and only once!
        # For the impact and collision event, we only store the event type.
        # _sub_event = np.zeros(num_spheres)
        if len(temp) > 0:
            _sub_event[list(temp.keys())] = [COLLISION_EVENT if temp[_id]["type"] == "collision" else
                                             IMPACT_EVENT for _id in temp.keys()]
        event_recorder.append(_sub_event)

    # ready for return
    spheres_location2 = frame_start_loc
    spheres_new_speeds = _start_speed

    return spheres_location2, spheres_new_speeds, (loc_recorder, speed_recorder, event_recorder)
