import numpy as np

"""Declare useful normal vectors"""
X_ = np.array([1, 0, 0])
Y_ = np.array([0, 1, 0])
Z_ = np.array([0, 0, 1])

Y_Z_NORM_OUT = np.array([1, 0, 0])
Y_Z_NORM_IN = np.array([-1, 0, 0])
X_Z_NORM_OUT = np.array([0, 1, 0])
X_Z_NORM_IN = np.array([0, -1, 0])
X_Y_NORM_OUT = np.array([0, 0, 1])
X_Y_NORM_IN = np.array([0, 0, -1])

AXIS_ = np.array([X_, Y_, Z_])
AXIS_NORM = np.linalg.norm(AXIS_, axis=1)
AXIS_NAME = ["x", "y", "z"]
AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def getLineSegment3D(
        start: np.ndarray,
        end: np.ndarray):
    """
    Get the line segment representation in Numpy. Also accept a batch of start points and end points as input.
    Args:
        start: a numpy array denoting the starting point in 3D.
        end:  a numpy array denoting the starting point in 3D.

    Returns:
        a tuple contains the starting point, end point, direction vector, and the normalized direction vector
        of the line segment.
    """
    assert isinstance(start, np.ndarray)
    assert isinstance(end, np.ndarray)
    assert start.shape[0] == 3
    assert end.shape[0] == 3

    direction_vector = end - start
    norm = np.linalg.norm(direction_vector)
    normalized_direction = direction_vector / norm

    return start, end, direction_vector, normalized_direction, norm


def helper_distance_point_to_line_segment(
        point: np.ndarray,
        segment_start: np.ndarray,
        segment_end: np.ndarray,
        line_segment_norm: float,
        line_segment_unit: np.ndarray):
    point_vector = point - segment_start

    if line_segment_norm == 0:
        # Degenerate case: line segment has zero length, so point is on the segment
        return np.linalg.norm(point_vector)

    # Project the point vector onto the line segment vector
    projection = np.dot(point_vector, line_segment_unit) * line_segment_unit
    # Determine if the closest point lies within the line segment or on the ray extending the line segment
    t = np.dot(projection, line_segment_unit)

    if t <= 0:
        # Closest point is before the start of the line segment
        return np.linalg.norm(point_vector)

    if t >= line_segment_norm:
        # Closest point is after the end of the line segment
        return np.linalg.norm(point - segment_end)

    # Closest point lies within the line segment
    closest_point = segment_start + t * line_segment_unit
    return np.linalg.norm(point - closest_point)


def helper_interpolate_line_segment_special(
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        const_coords: dict):
    """
    Interpolate given line segment between its start and end with information
    from const coords dictionary.Here we are assume given const coords indicate
    an axis.
    *This function should only be called during testing
    or from other functions in this script.*
    Args:
        seg_start: segment start
        seg_end: segment end
        const_coords: const coords dictionary.

    Returns:
        Interpolation point coordinates if found, None otherwise.
    """
    # always ensure the result is valid
    assert len(const_coords) == 2
    dir_seg = seg_end - seg_start
    res = []

    # find a valid interpolation direction and interpolate on that direction if valid
    for const_axis in const_coords.keys():
        # retrieve the const axis and its index
        idx = AXIS_IDX[const_axis]
        # print("const axis: ", const_axis)
        axis_projection = const_coords[const_axis] - seg_start[idx]
        seg_projection = dir_seg[idx]
        # print("axis projection: ", axis_projection)
        # print("seg projection: ", seg_projection)
        if seg_projection != 0:
            t = axis_projection / seg_projection
            # print("t: ", t)
            if 0 <= t <= 1:
                inter_point = seg_start + t * dir_seg
                res.append(inter_point)
        else:
            # not valid, try the next one
            continue

    # print("res: ", res)
    # check if all requirements are satisfied
    if len(res) > 0:
        for p in res:
            for axis in const_coords.keys():
                if p[AXIS_IDX[axis]] != const_coords[axis]:
                    return None
        return res[0]
    else:
        return None


def helper_line_intersection_axis_special(
        seg1_start: np.ndarray,
        seg1_end: np.ndarray,
        seg1_axis_check,
        seg2_start: np.ndarray,
        seg2_end: np.ndarray,
        seg2_axis_check):
    """
    Given line segments and their check axis results, determine if they intersect and solve
    the intersection if there is one.
    *This is a specialized function that only handles limited cases.*
    *Should only be called during testing or from other functions in this script.*
    Args:
        seg1_start: segment 1 start coordinates
        seg1_end: segment 1 end coordinates
        seg1_axis_check: segment 1 axis check result, a boolean list.
        seg2_start: segment 2 start coordinates
        seg2_end: segment 2 end coordinates
        seg2_axis_check: segment 2 axis check result, a boolean list.

    Returns:
        Intersection point coordinates if found, None otherwise.
    """
    # From the check axis result, determine and extract constant coords
    # For a line segment parallel to a given axis,
    # it would have constant coordinates on the other two axis,
    # and the intersection, if exists, should have same constant coordinates.
    # todo: always ensure the result is valid
    seg1_const_coords = {}
    if np.any(seg1_axis_check):
        for idx in range(3):
            if not seg1_axis_check[idx]:
                assert seg1_start[idx] == seg1_end[idx]
                seg1_const_coords[AXIS_NAME[idx]] = seg1_start[idx]

    seg2_const_coords = {}
    if np.any(seg2_axis_check):
        for idx in range(3):
            if not seg2_axis_check[idx]:
                assert seg2_start[idx] == seg2_end[idx]
                seg2_const_coords[AXIS_NAME[idx]] = seg2_start[idx]
    # print(f"==seg1: {[seg1_start, seg1_end]}, const: {seg1_const_coords}\n"
    #       f"==seg2: {[seg2_start, seg2_end]}, const: {seg2_const_coords}\n")
    # in the case of both segments are parallel to the axis,
    # they are either parallel or perpendicular
    if len(seg1_const_coords) == 2 and len(seg2_const_coords) == 2:
        # use dot product between boolean array to determine if parallel
        if np.dot(seg1_axis_check, seg2_axis_check):
            # parallel
            return None
        else:
            # perpendicular, just merge the const dict
            # check if the merged point is valid
            res_dict = {**seg1_const_coords, **seg2_const_coords}
            inter_point = np.array(
                [res_dict[AXIS_NAME[idx]] for idx in range(3)]
            )
            within_segments = np.all(
                (np.dot((seg1_start - inter_point), (seg1_end - inter_point)) <= 0) &
                (np.dot((seg2_start - inter_point), (seg2_end - inter_point)) <= 0)
            )
            if within_segments:
                return inter_point
            else:
                return None

    else:
        # one of them is parallel to the axis
        if len(seg1_const_coords) == 2:
            # interpolate seg2
            inter_point = helper_interpolate_line_segment_special(seg2_start, seg2_end, seg1_const_coords)
            # print(f"==seg1: {[seg1_start, seg1_end]} parallel to axis, inter point: {inter_point}")
            if inter_point is not None:
                within_segments = np.all(
                    (np.dot((seg1_start - inter_point), (seg1_end - inter_point)) <= 0) &
                    (np.dot((seg2_start - inter_point), (seg2_end - inter_point)) <= 0)
                )
                # print(f"seg1: {[seg1_start, seg1_end]} parallel to axis, inter point: {inter_point}"
                #       f"within segments: {within_segments}")
                if within_segments:
                    return inter_point
            else:
                return None
        elif len(seg2_const_coords) == 2:
            # interpolate seg1
            inter_point = helper_interpolate_line_segment_special(seg1_start, seg1_end, seg2_const_coords)
            # print(f"==seg2: {[seg2_start, seg2_end]} parallel to axis, inter point: {inter_point}")
            # print(f"===seg1: {[seg1_start, seg1_end]}, const: {seg1_const_coords}\n"
            #       f"===seg2: {[seg2_start, seg2_end]}, const: {seg2_const_coords}\n")
            if inter_point is not None:
                within_segments = np.all(
                    (np.dot((seg1_start - inter_point), (seg1_end - inter_point)) <= 0) &
                    (np.dot((seg2_start - inter_point), (seg2_end - inter_point)) <= 0)
                )
                if within_segments:
                    return inter_point
            else:
                return None


def helper_find_line_edge_intersection(
        seg1_start: np.ndarray,
        seg1_end: np.ndarray,
        seg1_unit_dir: np.ndarray,
        seg2_start: np.ndarray,
        seg2_end: np.ndarray,
        seg2_unit_dir: np.ndarray):
    """
    Quick routine for finding intersection between a line and axis.
    Assume that one of the line segment is parallel to axis, which is always
    true in the case of line and surface edge intersection.
    Args:
        seg1_start: segment 1 start
        seg1_end: segment 1 end
        seg1_unit_dir: segment 1 unit direction vector
        seg2_start: segment 2 start
        seg2_end: segment 2 end
        seg2_unit_dir: segment 2 unit direction vector

    Returns:

    """
    # todo: try optimize the following list comprehension...
    # todo: use array computation instead...

    # dir_seg_1_axis_check = [True if np.all(np.equal(np.cross(seg1_unit_dir, AXIS_[i]), np.zeros(3)))
    #                         else False for i in range(3)]

    # since inputs are unit vectors, just use 1 for their norms. No need to re-compute again.
    dir_seg_1_axis_check = np.isclose(np.dot(seg1_unit_dir, AXIS_.T), 1 * AXIS_NORM)

    # dir_seg_2_axis_check = [True if np.all(np.equal(np.cross(seg2_unit_dir, AXIS_[i]), np.zeros(3)))
    #                         else False for i in range(3)]

    dir_seg_2_axis_check = np.isclose(np.dot(seg2_unit_dir, AXIS_.T), 1 * AXIS_NORM)

    # print(f"--seg1: {[seg1_start, seg1_end]}, axis check: {dir_seg_1_axis_check}\n")
    # print(f"--seg2: {[seg2_start, seg2_end]}, axis check: {dir_seg_2_axis_check}\n")
    if np.any(dir_seg_1_axis_check) or np.any(dir_seg_2_axis_check):
        # print("++invoke helper!\n")
        return helper_line_intersection_axis_special(seg1_start, seg1_end, dir_seg_1_axis_check,
                                                     seg2_start, seg2_end, dir_seg_2_axis_check
                                                     )
    else:
        return None


def helper_derive_surface_boundary(
        surface_height: int,
        surface_centroid: np.ndarray,
        surface_type: str,
        mode="buildings",
        city_size=None):
    x = surface_centroid[0]
    y = surface_centroid[1]
    z = surface_centroid[2]

    if mode == "buildings":

        x_min = x - 0.5
        x_max = x + 0.5
        y_min = y - 0.5
        y_max = y + 0.5
        z_min = z - surface_height / 2
        z_max = z + surface_height / 2
    else:
        assert city_size is not None

        x_min = x - 0.5 * city_size
        x_max = x + 0.5 * city_size
        y_min = y - 0.5 * city_size
        y_max = y + 0.5 * city_size
        z_min = z - surface_height / 2
        z_max = z + surface_height / 2

    if surface_type == "x-y":
        return {
            "x": np.array([x_min, x_max]),
            "y": np.array([y_min, y_max]),
            "z": np.array([z, z])
        }

    elif surface_type == "x-z":
        return {
            "x": np.array([x_min, x_max]),
            "y": np.array([y, y]),
            "z": np.array([z_min, z_max])
        }
    elif surface_type == "y-z":
        return {
            "x": np.array([x, x]),
            "y": np.array([y_min, y_max]),
            "z": np.array([z_min, z_max])
        }
    else:
        raise ValueError("Invalid surface type")


def helper_find_surface_edges(
        surface_type: str,
        surface_boundary: dict):
    x_min = surface_boundary["x"][0]
    x_max = surface_boundary["x"][1]
    y_min = surface_boundary["y"][0]
    y_max = surface_boundary["y"][1]
    z_min = surface_boundary["z"][0]
    z_max = surface_boundary["z"][1]

    if surface_type == "y-z":
        corner_1 = np.array((x_min, y_min, z_min))
        corner_2 = np.array((x_min, y_max, z_min))
        corner_3 = np.array((x_min, y_min, z_max))
        corner_4 = np.array((x_min, y_max, z_max))
    elif surface_type == "x-z":

        corner_1 = np.array((x_min, y_min, z_min))
        corner_2 = np.array((x_max, y_min, z_min))
        corner_3 = np.array((x_max, y_min, z_max))
        corner_4 = np.array((x_min, y_min, z_max))
    elif surface_type == "x-y":
        corner_1 = np.array((x_min, y_min, z_min))
        corner_2 = np.array((x_max, y_min, z_min))
        corner_3 = np.array((x_max, y_max, z_min))
        corner_4 = np.array((x_min, y_max, z_min))
    else:
        raise ValueError("Invalid surface normal")

    edges = [np.array([corner_1, corner_2]),
             np.array([corner_2, corner_3]),
             np.array([corner_3, corner_4]),
             np.array([corner_4, corner_1])
             ]

    return edges


# todo: new function for finding AABB
def find_aabb_for_sphere(
        sphere_loc1: np.ndarray,
        sphere_loc2: np.ndarray,
        sphere_radius):
    """
    Get the axis-aligned-bounding-box for sphere's trajectory, use in early collision detection.
    Assume the sphere follows a linear motion.
    Args:
        sphere_loc1: a batch of start coordinates for the spheres.
        sphere_loc2: a batch end coordinates for the sphere.
        sphere_radius: sphere radius, could be value or a numpy array.

    Returns:
        A numpy array pf bounding box's center coordinates, and a numpy array of box dimension
        in x, y z axis represented by a numpy array.
    """

    # Modify this function so that we can handle a batch of spheres with Numpy methods on arrays directly

    # First calculate the bounding box dimension for all spheres
    bounding_box_dim = np.abs(sphere_loc1 - sphere_loc2) + 2.0 * sphere_radius

    # Then calculate the distance vector to box centroid
    vec_to_centroid = (bounding_box_dim - 2.0 * sphere_radius) / 2.0

    # Construct a stacked array for following operation

    # For both sphere locations, try opposite operation
    loc1_res1 = sphere_loc1 + vec_to_centroid
    loc1_res2 = sphere_loc1 - vec_to_centroid
    loc2_res1 = sphere_loc2 - vec_to_centroid
    loc2_res2 = sphere_loc2 + vec_to_centroid

    pair_1 = np.swapaxes(np.stack((loc1_res1, loc2_res1), axis=1), 1, 2)
    pair_2 = np.swapaxes(np.stack((loc1_res2, loc2_res2), axis=1), 1, 2)
    # print(f"\n--loc_1: {sphere_loc1}, loc_2: {sphere_loc2}")
    # print(f"--pair1: \n{pair_1}\n"
    #       f"--pair2: \n{pair_2}\n")
    box_centroids = []
    # if both coords in the pair are same, this is a valid box centroid
    # else, the box centroid is at the other pair arrays.
    # print(f"--pair shape: {pair_1.shape}")
    # print(f"--swap axis: bef: \n{pair_1}\n, aft: \n{np.swapaxes(pair_1, 1, 2)}"
    for i in range(pair_1.shape[0]):    # loop over each AABB
        s_aabb_centroid = []
        for j in range(3):
            # print(f"--checking: p1: {pair_1[i][j]}")
            # can not use direct "==" here, because of some numerical errors
            if np.all(np.isclose(pair_1[i][j][0], pair_1[i][j][1])):
                s_aabb_centroid.append(pair_1[i][j][0])
            else:
                s_aabb_centroid.append(pair_2[i][j][0])
        s_aabb_centroid = np.array(s_aabb_centroid)
        box_centroids.append(s_aabb_centroid)

    box_centroids = np.array(box_centroids)
    # print(f"--return: {box_centroids}")
    return box_centroids, bounding_box_dim


def helper_find_AABB_for_buildings(
        city: np.array):
    """
    Given a numpy array representing the city, return a list of AABB representation
    of all the buildings inside the city.
    Args:
        city: a 2D numpy array representation of the city.

    Returns:
        A dict of AABB representation  dict ("centroid", "dimensions") of all buildings inside the city.

    """
    indices = np.nonzero(city)

    buildings_height = city[indices]

    buildings_aabb_dict = {

        tuple(np.array([indices[0][i] + 0.5, indices[1][i] + 0.5, buildings_height[i] / 2.0])):
            [
                np.array([indices[0][i] + 0.5, indices[1][i] + 0.5, buildings_height[i] / 2.0]),
                np.array([1, 1, buildings_height[i]])
            ]
        for i in range(len(buildings_height))
    }

    return buildings_aabb_dict


def compute_building_surfaces(buildings):
    """
    Given the city map, find all the walls present and compute the line-segment forms of them.
    Args:
        buildings: city map.

    Returns:
        A list of line-segment forms of the walls. Each wall is represented by a list of two elements.
        The first element is always the buildings_x, y, z info, and the second element is the neighboring_x, y, and z.

    """
    size = buildings.shape[0]
    surfaces = []

    for i in range(size):
        for j in range(size):
            building_height = buildings[i, j]

            if building_height > 0:
                # Top surface
                top_surface = np.array([[i, j, building_height], [i, j, building_height]])
                surfaces.append(top_surface)

                # Check neighboring cells
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_i = i + dx
                    new_j = j + dy

                    # Check if the neighbor is within the bounds of the terrain
                    if 0 <= new_i < size and 0 <= new_j < size:
                        neighbor_height = buildings[new_i, new_j]

                        # Calculate the height difference
                        height_difference = building_height - neighbor_height

                        # If the height difference is positive and the neighbor is not a building,
                        # it contributes to the surface
                        if height_difference > 0:
                            surface = np.array([[i, j, building_height], [new_i, new_j, neighbor_height]])
                            surfaces.append(surface)

    return surfaces


# todo: have updated this function to return AABB representation of the buildings as well.
def compute_surfaces_from_buildings(surfaces, option="dict"):
    """
    Given the surfaces in line-segment forms, get the surfaces' normal and their surface_centroid, as
    well as their height, surface type and surface boundary.
    Associate the surfaces to a certain building's top surface x-y centroid.

    Arg:
        surfaces: the output of compute_building_surfaces.A list of surfaces in line-segment form.
        option: specifying return option. Default to "dict". Other valid option includes "list" and "simple".
    Returns:
        (either of the following)
        buildings_info[building][surface_facing] = {
                "surface_normal": normal,
                "coming_side": coming_side,
                "surface_centroid": surface_centroid,
                "surface_boundary": surface_boundary,
                "surface_edges": surface_edges
            }
        buildings_aabb[building] = [
                building_centroid, [1, 1, building_height]
        ]
        - "list": A dictionary whose keys are building centroid,
        and value is a list of corresponding surface representation
        using surface normal, surface centroid, surface height, and a string denoting surface type.
        - "dict": A dictionary whose keys are building centroid, and value is a dict whose keys
        are surface type in string, and keys are surface information.
        - "simple": A plain list contains a list with surface normal, centroid, height, and surface type.
    """
    assert option == "dict" or "list" or "simple"

    buildings_info = {}
    buildings_aabb = {}
    if option == "simple":
        buildings_info = []

    for surface in surfaces:
        centroid_x, centroid_y, building_height = surface[0]
        # print(f"surface 0: {surface[0]}")
        building = (centroid_x + 0.5, centroid_y + 0.5, building_height / 2.0)
        # print(f"building centroid: {building}")
        if option == "dict":
            if building not in buildings_info.keys():
                # add in new type of surface for bottom, using the same top surface representation.
                buildings_info[building] = {}
                # add in new default type of surface for bottom, using the top surface representation.
                surface_facing = "bottom"
                surface_type = "x-y"
                surface_centroid = np.array([centroid_x + 0.5, centroid_y + 0.5, 0])
                normal = np.array([0, 0, 1])
                surface_boundary = helper_derive_surface_boundary(0, surface_centroid, surface_type)
                surface_edges = helper_find_surface_edges(surface_type, surface_boundary)
                buildings_info[building][surface_facing] = {
                    "surface_normal": normal,
                    "coming_side": surface_type,
                    "surface_centroid": surface_centroid,
                    "surface_boundary": surface_boundary,
                    "surface_edges": surface_edges
                }

                buildings_aabb[building] = [np.array(building), np.array([1, 1, building_height])]
        elif option == "list":
            if building not in buildings_info.keys():
                buildings_info[building] = []
                buildings_aabb[building] = [np.array(building), np.array([1, 1, building_height])]

        element_x, element_y, element_z = surface[1]
        surface_height = building_height - element_z
        assert surface_height >= 0
        block_difference = (centroid_x - element_x, centroid_y - element_y)
        if building_height == element_z:
            surface_facing = "top"
            surface_type = "x-y"
            surface_centroid = np.array([centroid_x + 0.5, centroid_y + 0.5, building_height])
            # normal = np.array([0, 0, 1])
        elif block_difference[0] < 0 and block_difference[1] == 0:
            surface_facing = "up"
            surface_type = "y-z"
            surface_centroid = np.array([centroid_x, centroid_y + 0.5, element_z + surface_height / 2])
            # normal = np.array([-1, 0, 0])
        elif block_difference[0] > 0 and block_difference[1] == 0:
            surface_facing = "down"
            surface_type = "y-z"
            surface_centroid = np.array([centroid_x + 1, centroid_y + 0.5, element_z + surface_height / 2])
            # normal = np.array([1, 0, 0])
        elif block_difference[1] > 0 and block_difference[0] == 0:
            surface_facing = "left"
            surface_type = "x-z"
            surface_centroid = np.array([centroid_x + 0.5, centroid_y, element_z + surface_height / 2])
            # normal = np.array([0, 1, 0])
        elif block_difference[1] < 0 and block_difference[0] == 0:
            surface_facing = "right"
            surface_type = "x-z"
            surface_centroid = np.array([centroid_x + 0.5, centroid_y + 1, element_z + surface_height / 2])
            # normal = np.array([0, -1, 0])
        else:
            raise ValueError("Invalid surface type")

        # add in new type of surface for bottom, using the same top surface representation.

        # compute the surface boundary and edges alongside as well
        surface_boundary = helper_derive_surface_boundary(surface_height, surface_centroid, surface_type)
        surface_edges = helper_find_surface_edges(surface_type, surface_boundary)

        if option == "dict":
            buildings_info[building][surface_facing] = {
                # "surface_normal": normal,
                "coming_side": surface_type,
                "surface_centroid": surface_centroid,
                "surface_boundary": surface_boundary,
                "surface_edges": surface_edges
            }
        elif option == "list":
            buildings_info[building].append(
                [surface_type, surface_centroid, surface_boundary, surface_edges]
            )
        else:
            buildings_info.append(
                [surface_type, surface_centroid, surface_boundary, surface_edges]
            )

    return buildings_info, buildings_aabb


def compute_city_borders(
        city_size: int,
        city_height: int):
    """
    Treat city borders as buildings.
    Compute the city borders information as if there are walls around the city.
    Coordinate system has x-axis pointing to south, y-axis pointing to east.
    Args:
        city_size: size of the city. Assume a square city.
        city_height: height of the city.

    Returns:
        A dictionary whose keys are imaginary city wall centroids, and values
        being the wall surfaces and edges information.

    """
    # compute wall centroid: N-S-W-E-Top-Bottom orders
    margin = 1e-4
    h = city_height
    a = city_size
    w = 10  # wall or border's width

    # point outside city boundary.
    # border_normal = np.array([
    #     Y_Z_NORM_IN,  # North
    #     Y_Z_NORM_OUT,  # South
    #     X_Z_NORM_IN,  # West
    #     X_Z_NORM_OUT,  # East
    #     X_Y_NORM_OUT,  # Top
    #     X_Y_NORM_IN  # Bottom
    # ])

    # shift the centroids slightly out of the ideal borders
    # to avoid simultaneous impacts with two walls ...
    b_centroids = np.array([
        np.array([-w / 2, a / 2, h / 2]),  # North
        np.array([a + w / 2, a / 2, h / 2]),  # South
        np.array([a / 2, -w / 2, h / 2]),  # West
        np.array([a / 2, a + w / 2, h / 2]),  # East
        np.array([a / 2, a / 2, h + w / 2]),  # TOP
        np.array([a / 2, a / 2, -w / 2])  # BOTTOM
    ])

    s_centroids = np.array([
        np.array([0, a / 2, h / 2]),  # North
        np.array([a, a / 2, h / 2]),  # South
        np.array([a / 2, 0, h / 2]),  # West
        np.array([a / 2, a, h / 2]),  # East
        np.array([a / 2, a / 2, h]),  # TOP
        np.array([a / 2, a / 2, 0])  # BOTTOM
    ])
    # dim: x, y, z length of the bounding box
    dimensions = [
        np.array([w, a, h]),  # North
        np.array([w, a, h]),  # South
        np.array([a, w, h]),  # West
        np.array([a, w, h]),  # East
        np.array([a, a, w]),  # Top
        np.array([a, a, w])  # Bottom
    ]

    surface_type = [
        "y-z", "y-z", "x-z", "x-z", "x-y", "x-y"
    ]
    # create fake surfaces for the buildings...from centroids and dimensions
    # the surface is represented by surface centroids, surface normal, and edges

    # cast to list for following ops
    b_centroids = list(b_centroids)
    # border_normal = list(border_normal)

    # compute wall edges
    boundaries = [
        helper_derive_surface_boundary(city_height, s_centroids[i], surface_type[i], "city", city_size)
        for i in range(len(s_centroids))
    ]
    # print(f"----boundaries:")
    # for i in range(len(boundaries)):
    #     print(f"-----{boundaries[i]}")

    edges = [
        helper_find_surface_edges(surface_type[j], boundaries[j])
        for j in range(len(b_centroids))
    ]

    # Face inside city?
    surface_facing = [
        "down",  # North
        "up",  # South
        "right",  # West
        "left",  # East
        "bottom",  # top
        "top"  # bottom
    ]
    # todo: new surface facing: bottom!
    borders_info = {
        tuple(b_centroids[k]): {
            surface_facing[k]:
                {
                 "coming_side": surface_type[k],
                 "surface_centroid": s_centroids[k],
                 "surface_boundary": boundaries[k],
                 "surface_edges": edges[k]}
        } for k in range(len(s_centroids))
    }

    borders_aabb = {
        tuple(b_centroids[k]): [b_centroids[k], dimensions[k]]
        for k in range(len(b_centroids))
    }
    # print(f"--borders info: \n{borders_info}\n")
    return borders_info, borders_aabb


def compute_city_info(
        city_height: int,
        city: np.ndarray):
    """
    Given the information for city, compute and compile all necessary auxiliary information about
    the city or buildings into dictionary for one-off access. Rely on relevant helper and sub functions
    in geometry.
    Args:
        city_height: height of the city.
        city: a 2D numpy array representation of the city.

    Returns:
        a dictionary with all necessary city information.

    """
    city_size = city.shape[0]

    borders_info, borders_aabb = compute_city_borders(city_size, city_height)
    buildings_surfaces = compute_building_surfaces(city)
    buildings_info, buildings_aabb = compute_surfaces_from_buildings(buildings_surfaces)

    # merge aabb and all "buildings" information
    assert isinstance(borders_aabb, dict)
    assert isinstance(borders_info, dict)
    assert isinstance(buildings_aabb, dict)
    assert isinstance(buildings_info, dict)

    city_aabb = {**borders_aabb, **buildings_aabb}
    city_info = {**borders_info, **buildings_info}

    return city_info, city_aabb


# todo: test this!
def check_aabb_intersection(
        aabb1: list,
        aabb2: list):
    """
    Check if any intersection between given bounding boxes.
    Surface contact does not count as intersection.
    Args:
        aabb1:
        aabb2:

    Returns:

    """
    center1, dimensions1 = aabb1
    center2, dimensions2 = aabb2

    half_dimensions1 = dimensions1 / 2.0
    half_dimensions2 = dimensions2 / 2.0

    # Calculate min and max coordinates of each AABB
    min_coords1 = center1 - half_dimensions1
    max_coords1 = center1 + half_dimensions1
    min_coords2 = center2 - half_dimensions2
    max_coords2 = center2 + half_dimensions2

    intersect = np.logical_or.reduce((
        max_coords1 <= min_coords2,
        min_coords1 >= max_coords2
    ))

    return not np.any(intersect)


def find_line_intersection(
        seg1_start: np.ndarray,
        seg1_end: np.ndarray,
        seg2_start: np.ndarray,
        seg2_end: np.ndarray):
    """
    Find the intersection point between two line segments. Call specialized functions
    for edge cases, like those involving horizontal/vertical lines.
    Args:
        seg1_start: segment 1 start coordinates.
        seg1_end: segment 1 end coordinates.
        seg2_start: segment 2 start coordinates.
        seg2_end: segment 2 end coordinates.

    Returns:
        Intersection point coordinates if found, None otherwise.
    """
    dir_seg1 = seg1_end - seg1_start
    dir_seg2 = seg2_end - seg2_start

    # seg1_norm = np.linalg.norm(dir_seg1)
    # seg2_norm = np.linalg.norm(dir_seg2)

    # check if these segments are parallel to axis
    # True if parallel to the axis, False otherwise
    dir_seg_1_axis_check = [~np.any(np.cross(dir_seg1, AXIS_[i])) for i in range(3)]

    dir_seg_2_axis_check = [~np.any(np.cross(dir_seg2, AXIS_[i])) for i in range(3)]

    # let specialized functions handle the case, as general solution will be problematic
    if np.any(dir_seg_1_axis_check) or np.any(dir_seg_2_axis_check):
        # print("pass to helper")
        return helper_line_intersection_axis_special(seg1_start, seg1_end, dir_seg_1_axis_check, seg2_start, seg2_end,
                                                     dir_seg_2_axis_check)

    cross_prod = np.cross(dir_seg1, dir_seg2)

    if np.allclose(cross_prod, 0):
        return None  # No intersection exists

    with np.errstate(divide='ignore', invalid='ignore'):
        delta_start = seg2_start - seg1_start
        t = np.divide(np.cross(delta_start, dir_seg2), cross_prod)
        u = np.divide(np.cross(delta_start, dir_seg1), cross_prod)

    # Check if both parameters t and u are within the valid range [0, 1]
    valid_intersection = np.logical_and(
        np.logical_and(t >= 0, t <= 1), np.logical_and(u >= 0, u <= 1))

    if np.any(valid_intersection):
        intersection_point = seg1_start + t[valid_intersection] * dir_seg1
        return intersection_point
    else:
        return None  # No intersection within the line segments


def find_line_surface_intersection(
        line_start: np.ndarray,
        line_end: np.ndarray,
        surface_normal: np.ndarray,
        surface_centroid: np.ndarray):
    """
    An optimized way to solve the intersection problem, exploiting the fact that all surfaces in questions
    are parallel to one of the x-y, x-z, y-z planes.
    Args:
        line_start: start coordinates of the line segment.
        line_end: end coordinates of the line segment.
        surface_normal: surface normal vector.
        surface_centroid: surface centroid coordinate.

    Raises:
        Invalid surface normal:
    Returns:
        Intersection point coordinates if intersection exists, else returns None.
    """
    # Determine the surface type based on the surface normal
    if np.dot(surface_normal, X_) != 0:
        surface_type = "y-z"
    elif np.dot(surface_normal, Y_) != 0:
        surface_type = "x-z"
    elif np.dot(surface_normal, Z_) != 0:
        surface_type = "x-y"
    else:
        raise ValueError("Invalid surface normal")

    # Extract the constant coordinate value based on the surface type
    if surface_type == "x-z":
        constant_coord = surface_centroid[1]
    elif surface_type == "y-z":
        constant_coord = surface_centroid[0]
    elif surface_type == "x-y":
        constant_coord = surface_centroid[2]
    else:
        constant_coord = None

    # Compute the intersection point based on the surface type and constant coordinate
    with np.errstate(divide='ignore', invalid='ignore'):
        if surface_type == "x-z":
            t = (constant_coord - line_start[1]) / (line_end[1] - line_start[1])
            intersection_point = line_start + t * (line_end - line_start)
            intersection_point[1] = constant_coord
        elif surface_type == "y-z":
            t = (constant_coord - line_start[0]) / (line_end[0] - line_start[0])
            intersection_point = line_start + t * (line_end - line_start)
            intersection_point[0] = constant_coord
        elif surface_type == "x-y":
            t = (constant_coord - line_start[2]) / (line_end[2] - line_start[2])
            intersection_point = line_start + t * (line_end - line_start)
            intersection_point[2] = constant_coord
        else:
            raise ValueError("Invalid surface type")

    # Check if the intersection point lies within the line segment
    valid_t = np.logical_and(0 <= t, t <= 1)
    if np.all(valid_t):
        return intersection_point
    else:
        return None


def find_sphere_surface_contact(
        sphere_radius: float,
        line_start: np.ndarray,
        line_end: np.ndarray,
        surface_normal: np.ndarray,
        surface_centroid: np.ndarray):
    """
    An optimized way to find the sphere center coordinates when it traverses on given line segment
    and contact the surface, exploiting the fact that all surfaces in questions
    are parallel to one of the x-y, x-z, y-z planes.
    Args:
        sphere_radius: radius of the sphere.
        line_start: start coordinates of line segment.
        line_end: and coordinate of line segment.
        surface_normal: surface normal vector. Should be pointing outwards.
        surface_centroid:surface centroid.

    Returns:
        The sphere center coordinate when contact the surface, and its relative position on the line segment.

    """
    # Determine the surface type based on the surface normal
    if np.dot(surface_normal, X_) != 0:
        surface_type = "y-z"
    elif np.dot(surface_normal, Y_) != 0:
        surface_type = "x-z"
    elif np.dot(surface_normal, Z_) != 0:
        surface_type = "x-y"
    else:
        raise ValueError("Invalid surface normal")

    # Extract the constant coordinate value based on the surface type
    # Move the surface closer to the line along normal direction, by a distance of sphere radius
    if surface_type == "x-z":
        constant_coord = surface_centroid[1] + sphere_radius * surface_normal[1]
    elif surface_type == "y-z":
        constant_coord = surface_centroid[0] + sphere_radius * surface_normal[0]
    elif surface_type == "x-y":
        constant_coord = surface_centroid[2] + sphere_radius * surface_normal[2]
    else:
        constant_coord = None

    # Compute the intersection point based on the surface type and constant coordinate
    with np.errstate(divide='ignore', invalid='ignore'):
        if surface_type == "x-z":
            t = (constant_coord - line_start[1]) / (line_end[1] - line_start[1])
            intersection_point = line_start + t * (line_end - line_start)
            intersection_point[1] = constant_coord
        elif surface_type == "y-z":
            t = (constant_coord - line_start[0]) / (line_end[0] - line_start[0])
            intersection_point = line_start + t * (line_end - line_start)
            intersection_point[0] = constant_coord
        elif surface_type == "x-y":
            t = (constant_coord - line_start[2]) / (line_end[2] - line_start[2])
            intersection_point = line_start + t * (line_end - line_start)
            intersection_point[2] = constant_coord
        else:
            raise ValueError("Invalid surface type")

    # Check if the intersection point lies within the line segment
    valid_t = np.logical_and(0 <= t, t <= 1)
    if np.all(valid_t):
        return intersection_point, t
    else:
        return None


# this function does not produce desired outputs...
# the logic might be correct, but getting desired output of closest sphere position is still expensive
# and need to check for corner case as well.
# no better than the next function that does expected things.
# def new_find_closest_position_sphere_and_edge(
#         sphere_start: np.ndarray,
#         sphere_end: np.ndarray,
#         radius: float,
#         edge_start: np.ndarray,
#         edge_end: np.ndarray):
#
#     direction = sphere_end - sphere_start
#
#     # Calculate relative position vector of edge's start point with respect to sphere's start point
#     relative_position = edge_start - sphere_start
#
#     # Calculate dot product of direction vector and relative position vector
#     dot_product = np.dot(direction, relative_position)
#
#     if dot_product < 0:
#         # Edge is behind the sphere, no collision
#         print(f"---Edge is behind the sphere, no collision!")
#         return None
#
#         # Calculate magnitude of direction vector
#     direction_magnitude = np.linalg.norm(direction)
#
#     # Calculate magnitude of direction vector
#     direction_magnitude = np.linalg.norm(direction)
#
#     # Calculate squared distance from start point of sphere to the closest point on edge
#     t = dot_product / direction_magnitude ** 2
#     t = np.clip(t, 0, 1)  # Clamp t between 0 and 1 to ensure the closest point is on edge
#     closest_point = edge_start + t * (edge_end - edge_start)
#
#     squared_distance = np.linalg.norm(sphere_start - closest_point) ** 2
#     print(f"---dis norm: {np.linalg.norm(sphere_start - closest_point)}, s_start: {sphere_start}")
#     # Check if there is a collision
#     # is_collision = np.is_close(squared_distance)
#     if squared_distance <= radius ** 2:
#         return closest_point, np.sqrt(squared_distance)
#
#     # No collision
#     print(f"---No collision! squared distance: {squared_distance}, closest pos: {closest_point}")
#     return None


# todo: fix this one as well
def find_closest_positions(
        segment1_start: np.ndarray,
        segment1_end: np.ndarray,
        segment2_start: np.ndarray,
        segment2_end: np.ndarray):
    """
    Find the closest positions between two line segments.
    Args:
        segment1_start:
        segment1_end:
        segment2_start:
        segment2_end:

    Returns:
        the closest positions on the two lines if such positions exist, else None.
    """
    # Calculate direction vectors for both line segments
    dir_segment1 = segment1_end - segment1_start
    dir_segment2 = segment2_end - segment2_start

    # Calculate the cross product of the direction vectors
    cross_product_norm = np.linalg.norm(np.cross(dir_segment1, dir_segment2))
    # print(f"cross prod: {cross_product_norm}")
    # Check if the cross product is zero (indicating parallel or collinear lines)
    if np.allclose(cross_product_norm, 0):
        # print("parallel or collinear")
        # If the lines are parallel or collinear, return None
        return None

    # Solve the system of linear equations
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_start = segment1_start - segment2_start
        a = np.dot(dir_segment1, dir_segment1)
        b = np.dot(dir_segment1, dir_segment2)
        c = np.dot(dir_segment2, dir_segment2)
        d = np.dot(dir_segment1, delta_start)
        e = np.dot(dir_segment2, delta_start)
        t = (b * e - c * d) / (cross_product_norm ** 2)
        u = (a * e - b * d) / (cross_product_norm ** 2)
        # t = np.cross(delta_start, dir_segment2) / cross_product_norm
        # u = np.cross(delta_start, dir_segment1) / cross_product_norm
        # print(f"cross prod 1: {np.cross(delta_start, dir_segment2)}\n")
        # print(f"t1 nominator: {(b * e - c * d)}\n")
        # print(f"cross prod 2: {np.cross(delta_start, dir_segment1)}\n")
        # print(f"t1 nominator: {(a * e - b * d)}\n")

    # print(f"t: {t}, u: {u}")
    # Check if both parameters t and u are within the valid range [0, 1]
    validity = np.logical_and((t >= 0) & (t <= 1), (u >= 0) & (u <= 1))

    # Calculate the closest positions on the line segments
    if np.all(validity):
        closest_segment1 = segment1_start + t * dir_segment1
        closest_segment2 = segment2_start + u * dir_segment2
        return closest_segment1, closest_segment2
    else:
        # print("Not valid")
        return None


def solve_sphere_collision(
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
        the collision position of sphere 1 and sphere 2 if a collision is possible,
        else None.
    """
    closest_result = find_closest_positions(start1, end1, start2, end2)

    if closest_result is None:
        # collinear or parallel case
        if np.linalg.norm(start1 - start2) <= radius1 + radius2:
            # collide at start
            return start1, start2
        else:
            # collide near end for collinear
            start_distance = np.linalg.norm(start1 - start2)
            clearance = radius2 + radius1
            approach_distance = start_distance - clearance
            approach_speed = np.linalg.norm(speed1 - speed2)
            approach_time = approach_distance / approach_speed

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

    collision_position_1 = start1 + approach_time * speed1
    collision_position_2 = start2 + approach_time * speed2

    return collision_position_1, collision_position_2
