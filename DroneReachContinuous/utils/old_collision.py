import numpy as np
from miscellaneous import *

float_dtype = np.float32
int_dtype = np.int32


def impact_check_first(
        num_agents_: int,
        city_map_: np.ndarray,
        x_paths: np.ndarray,
        y_paths: np.ndarray,
        z_paths: np.ndarray):
    """
    Check impact with building along an interpolated paths.
    This shall be called before checking collision with agents.
    Args:
        num_agents_: (int): number of agents.
        city_map_ (np.ndarray): a 2D numpy array for the city.
        x_paths (np.ndarray): interpolation of agents' paths in x-axis.
        y_paths (np.ndarray): interpolation of agents' paths in y-axis.
        z_paths (np.ndarray): interpolation of agents' paths in z-axis.

    Returns:
        a dictionary whose keys are agent ids involved in impact, values are a list with impact point coordinates.
    """
    # check collisions with buildings
    blk_x = np.array([int_dtype(coords_ - coords_ % 1) for coords_ in x_paths], dtype=int_dtype)
    blk_y = np.array([int_dtype(coords_ - coords_ % 1) for coords_ in y_paths], dtype=int_dtype)
    blk_z = np.array([int_dtype(coords_ - coords_ % 1) for coords_ in z_paths], dtype=int_dtype)
    # print("blk_z: ", blk_z)
    # print("z_paths: ", z_paths)
    # this dict would have agent id as keys, and value being a list whose
    # values being the check results (1 for collision, 0 for free)
    impact_with_buildings_res = {}
    # assert len(blk_paths) == num_agents
    full_path_len = blk_x.shape[1]
    for id_ in range(num_agents_):
        check_list = []
        # print("checking id:", id_)
        for blk_id in range(full_path_len):
            _x = blk_x[id_][blk_id]
            _y = blk_y[id_][blk_id]
            _z = blk_z[id_][blk_id]
            # print(f"checking path {_x}, {_y}, {_z}")
            if city_map_[_x][_y] > _z:
                # print("impacted at blk_id: ", blk_id)
                check_list.append(1)
                # no need to check further
                break
            else:
                check_list.append(0)
        impact_with_buildings_res[id_] = check_list

    impact_with_buildings = {}
    for id_ in range(num_agents_):
        if sum(impact_with_buildings_res[id_]) != 0:
            # collision detected
            # there is no way for the len == 1, where agent impacts without stepping
            assert len(impact_with_buildings_res[id_]) >= 2
            # if len == 1 need to back-track to previous timestep for pre-impact coords
            impact_idx = len(impact_with_buildings_res[id_]) - 1
            # print("impact idx: ", impact_idx)
            pre_impact_idx = impact_idx - 1  # useful for resolving
            pre_impact_coords = np.array([
                x_paths[id_][pre_impact_idx],
                y_paths[id_][pre_impact_idx],
                z_paths[id_][pre_impact_idx]
            ], dtype=float_dtype)
            impact_coords = np.array([
                x_paths[id_][impact_idx],
                y_paths[id_][impact_idx],
                z_paths[id_][impact_idx]
            ], dtype=float_dtype)
            impact_with_buildings[id_] = \
                [impact_coords, pre_impact_coords, impact_idx, full_path_len]

    return impact_with_buildings


def collision_check_then(
        _num_agents: int,
        collision_radius: float,
        impact_with_buildings: dict,
        x_paths: np.ndarray,
        y_paths: np.ndarray,
        z_paths: np.ndarray):
    """
    Check agents' collision with each other along their interpolated paths.
    Args:
        _num_agents (int): number of agents
        collision_radius (float): minimum distance for clearance between coords.
        impact_with_buildings (dict): dict contains impact information.
        x_paths (np.ndarray): interpolation of agents' paths in x-axis.
        y_paths (np.ndarray): interpolation of agents' paths in y-axis.
        z_paths (np.ndarray): interpolation of agents' paths in z-axis.

    Returns:
        the updated impact information, and a dictionary whose keys are agent ids involved in collision, values are a list with the other impact point agent ids, and index in paths.

    """
    checked_set = set()
    collision_with_agents_res = {}
    collision_res = {}
    collision_pair_dict = {}

    for ag_id in range(_num_agents):
        # init pairs and result lists in dict for new points
        pair = set()
        pair.add(ag_id)
        if ag_id not in collision_with_agents_res.keys():
            collision_with_agents_res[ag_id] = []
        # load the paths
        ag_paths = np.array(
            (x_paths[ag_id], y_paths[ag_id], z_paths[ag_id]),
            dtype=float_dtype
        ).transpose()
        paths_len = ag_paths.shape[0]

        # perform the check for new pairs of points
        for _id in range(_num_agents):
            pair.add(_id)
            if ag_id != _id and pair not in checked_set:
                # init the result lists as well
                collision_with_agents_res[_id] = []

                _id_paths = np.array(
                    (x_paths[_id], y_paths[_id], z_paths[_id]),
                    dtype=float_dtype
                ).transpose()
                _id_path_len = _id_paths.shape[0]

                # retrieve impact information for the pair
                if ag_id in impact_with_buildings.keys():
                    ag_id_impact_idx = impact_with_buildings[ag_id][0]
                    ag_id_impact = True
                else:
                    ag_id_impact_idx = np.inf
                    ag_id_impact = False

                if _id in impact_with_buildings.keys():
                    _id_impact_idx = impact_with_buildings[_id][0]
                    _id_impact = True
                else:
                    _id_impact_idx = np.inf
                    _id_impact = False

                check_list_ = []
                for _idx in range(min(paths_len, _id_path_len)):
                    distance = np.linalg.norm(
                        ag_paths[_idx] - _id_paths[_idx]
                    )
                    if distance <= collision_radius * 2:
                        # a collision detected
                        # no need to check for collision given a proceeding impact
                        if _idx <= ag_id_impact_idx and _id <= _id_impact_idx:
                            # valid collision, as impact time is at later or infinity
                            check_list_.append(-1)
                            collision_res[ag_id] = -1
                            collision_res[_id] = -1
                            collision_pair_dict[ag_id] = _id
                            # and nullify the impact later for the pair
                            if ag_id_impact:
                                impact_with_buildings.pop(ag_id)
                            if _id_impact:
                                impact_with_buildings.pop(_id)
                            # no need to check further
                            break
                        else:
                            # one of the pair impacts buildings earlier
                            break
                    else:
                        check_list_.append(0)
                # complete the check for one pair
                checked_set.add(pair)
                pair.remove(_id)
                # update dict for this pair of points
                collision_with_agents_res[ag_id].append(check_list_)
                collision_with_agents_res[_id].append(check_list_)
            else:
                pass

    collision_with_agents = {}
    for ag_id in collision_with_agents_res.keys():
        result = np.array(collision_with_agents_res[ag_id]).sum()
        assert result == -1 or 0
        for res in collision_with_agents_res[ag_id]:
            if sum(res) == -1:
                # need to retrieve the other point involved in collision as well
                # todo: what if len(res) == 1?
                if len(res) < 2:
                    ag_id_idx = len(res) - 1
                else:
                    ag_id_idx = len(res) - 2
                _id = collision_pair_dict[ag_id]
                _id_res_arr = np.array(collision_with_agents_res[_id])
                assert _id_res_arr.sum() == -1
                _id_idx = ag_id_idx
                ag_id_coords = np.array([
                    x_paths[ag_id][ag_id_idx],
                    y_paths[ag_id][ag_id_idx],
                    z_paths[ag_id][ag_id_idx]
                ], dtype=float_dtype)
                collision_with_agents[ag_id] = [_id, ag_id_coords]
                if _id not in collision_with_agents.keys():
                    # new
                    collision_with_agents_res.pop(_id)
                    _id_coords = np.array([
                        x_paths[_id][_id_idx],
                        y_paths[_id][_id_idx],
                        z_paths[_id][_id_idx]
                    ], dtype=float_dtype)
                    collision_with_agents[_id] = [ag_id, _id_coords]

    return impact_with_buildings, checked_set, collision_with_agents


def resolve_impact(
        num_agents: int,
        impact_damping: float,
        impact_with_buildings: dict,
        loc_x_curr_t: np.ndarray,
        loc_y_curr_t: np.ndarray,
        loc_z_curr_t: np.ndarray,
        speed_curr_t: np.ndarray,
        vert_speed_curr_t: np.ndarray,
        dir_curr_t: np.ndarray):
    """
    Flip agents current direction and reduce speeds by the damping ratio upon
    impact with buildings.
    Args:
        num_agents: total number of agents
        impact_damping: speed damping coefficient upon impact.
        impact_with_buildings: a dictionary with keys being impact agent ids.
        loc_x_curr_t: all agents current location in x-axis.
        loc_y_curr_t: all agents current location in y-axis
        loc_z_curr_t: all agents current location in z-axis
        speed_curr_t: all agents current planar speed
        vert_speed_curr_t: all agents current vertical speed
        dir_curr_t: all agents current directions.

    Returns:
        Updated speed, direction and locations for all agents.

    """
    assert loc_x_curr_t.shape[0] == num_agents
    assert loc_y_curr_t.shape[0] == num_agents
    assert loc_z_curr_t.shape[0] == num_agents
    assert speed_curr_t.shape[0] == num_agents
    assert vert_speed_curr_t.shape[0] == num_agents
    assert dir_curr_t.shape[0] == num_agents

    impact_agent_ids = [ag_id for ag_id in impact_with_buildings.keys()]

    impact_coords = np.array(
        [impact_with_buildings[ag_id][0] for ag_id in impact_agent_ids], dtype=float_dtype)

    pre_impact_coords = np.array(
        [impact_with_buildings[ag_id][1] for ag_id in impact_agent_ids], dtype=float_dtype)

    impact_idx = np.array(
        [impact_with_buildings[ag_id][2] for ag_id in impact_agent_ids], dtype=int_dtype)

    full_path_len = np.array(
        [impact_with_buildings[ag_id][3] for ag_id in impact_agent_ids], dtype=int_dtype)

    # get the borders crossed
    crossed_blk_borders = np.rint(pre_impact_coords, dtype=int_dtype)

    # mirror the coordinates along the crossed borders
    after_impact_coords = 2 * impact_coords - crossed_blk_borders

    # update the location
    loc_x_curr_t[impact_agent_ids] = after_impact_coords[0]
    loc_y_curr_t[impact_agent_ids] = after_impact_coords[1]
    loc_z_curr_t[impact_agent_ids] = after_impact_coords[2]

    loc_curr_t = (loc_x_curr_t, loc_y_curr_t, loc_z_curr_t)

    # damp the speed
    speed_curr_t[impact_agent_ids] = impact_damping * speed_curr_t[impact_agent_ids]
    vert_speed_curr_t[impact_agent_ids] = \
        impact_damping * vert_speed_curr_t[impact_agent_ids]

    speed_updated = (speed_curr_t, vert_speed_curr_t)
    # flip direction...
    x_dir = np.cos(dir_curr_t)
    y_dir = np.sin(dir_curr_t)

    x_dir[impact_agent_ids] = - x_dir[impact_agent_ids]
    y_dir[impact_agent_ids] = - y_dir[impact_agent_ids]

    # get back the direction in radian
    dir_curr_t = np.arctan2(x_dir, y_dir)

    return speed_updated, dir_curr_t, loc_curr_t


def resolve_collision(
        collision_pair_set: set,
        collision_with_agents: dict,
        speed_curr_t: np.ndarray,
        vert_speed_curr_t: np.ndarray,
        dir_curr_t: np.ndarray):
    agent_ids = [ag_id for ag_id in collision_with_agents.keys()]

    collision_pair_list = [tuple(pair) for pair in collision_pair_set]

    v_x = speed_curr_t[agent_ids] * np.cos(dir_curr_t[agent_ids])
    v_y = speed_curr_t[agent_ids] * np.sin(dir_curr_t[agent_ids])
    v_z = vert_speed_curr_t[agent_ids]

    # we would resolve the collision for each pair, instead of each agent.

    # for each pair of colliders, get their delta distances and velocities
    for pair in collision_pair_list:
        # determine the 3D angle between colliders
        _1_id = pair[0]
        _2_id = pair[1]

        _1_delta_r = np.array(
            collision_with_agents[_1_id][1] - collision_with_agents[_2_id][1],
            dtype=float_dtype)
        _2_delta_r = np.array(
            collision_with_agents[_2_id][1] - collision_with_agents[_1_id][1],
            dtype=float_dtype)

        _1_velocity = np.array(
            (v_x[_1_id], v_y[_1_id], v_z[_1_id]), dtype=float_dtype)
        _2_velocity = np.arrat(
            (v_x[_1_id], v_y[_1_id], v_z[_1_id]), dtype=float_dtype
        )

        _1_angle = np.arccos(
            np.dot(_1_velocity, _1_delta_r) /
            (np.linalg.norm(_1_velocity) * np.linalg.norm(_1_delta_r)),
            dtype=float_dtype)
        _2_angle = np.arccos(
            np.dot(_2_velocity, _2_delta_r) /
            (np.linalg.norm(_2_velocity) * np.linalg.norm(_2_delta_r)),
            dtype=float_dtype)

        # calculate colliders' velocity vectors towards each other
        _1_center_vel = _1_velocity * np.cos(_1_angle)
        _2_center_vel = _2_velocity * np.cos(_2_angle)

        # decompose vector into x-y-z components(?)

        # determine velocity normal to the center-line
        _1_normal_vel = _1_velocity - _1_center_vel
        _2_normal_vel = _2_velocity - _2_center_vel

        # switch the colliders' velocity vectors
        temp = _1_center_vel
        _1_center_vel = _2_center_vel
        _2_center_vel = temp

        # compose new vectors into new velocity
        _1_new_velocity = _1_normal_vel + _1_center_vel
        _2_new_velocity = _2_normal_vel + _2_center_vel

        # update direction, speed, vertical speed
        _1_new_dir = np.arctan2(_1_new_velocity[1], _1_new_velocity[0], dtype=float_dtype)
        _2_new_dir = np.arctan2(_2_new_velocity[1], _2_new_velocity[0], dtype=float_dtype)
        _1_new_speed = np.sqrt(
            _1_new_velocity[0] ** 2 + _1_new_velocity[1] ** 2, dtype=float_dtype
        )
        _2_new_speed = np.sqrt(
            _2_new_velocity[0] ** 2 + _2_new_velocity[1] ** 2, dtype=float_dtype
        )
        _1_new_v_speed = float_dtype(_1_new_velocity[2])
        _2_new_v_speed = float_dtype(_2_new_velocity[2])

        speed_curr_t[_1_id] = _1_new_speed
        speed_curr_t[_2_id] = _2_new_speed
        vert_speed_curr_t[_1_id] = _1_new_v_speed
        vert_speed_curr_t[_2_id] = _2_new_v_speed
        dir_curr_t[_1_id] = _1_new_dir
        dir_curr_t[_2_id] = _2_new_dir

        # it seems that we do not need to touch the location...
    return speed_curr_t, vert_speed_curr_t, dir_curr_t


def impact_and_collision(
        time_step: int,
        _num_agents_: int,
        city_map_: np.ndarray,
        global_state_: dict,
        loc_x_curr_t_: np.ndarray,
        loc_y_curr_t_: np.ndarray,
        loc_z_curr_t_: np.ndarray,
        speed_curr_t_: np.ndarray,
        vert_speed_curr_t_: np.ndarray,
        dir_curr_t_: np.ndarray,
        spacing_for_drones: float,
        collision_radius_over_spacing: float = 0.2):
    """
    Given the current time-step and all agent's state, check for impact with buildings and collision within agents.
    Args:
        collision_radius_over_spacing: ratio of collision radius over spacing for drones at spawn.
        spacing_for_drones: drones spacing when spawned.
        time_step: current time-step.
        _num_agents_: total number of agents
        city_map_: numpy 2D arrays with entries indicating buildings height.
        global_state_: global state dictionary that holds information for all agents at all timestep.
        loc_x_curr_t_: current agents x location waiting to check and update.
        loc_y_curr_t_: current agents y location waiting to check and update.
        loc_z_curr_t_: current agents z location waiting to check and update.
        speed_curr_t_: current agents planar speed waiting to check and update.
        vert_speed_curr_t_: current agents vertical speed waiting to check and update.
        dir_curr_t_: current agents direction waiting to check and update.

    Returns:
        Updated locations, kinematics, and agents ids involved in impact and collision.

    """
    # todo: might need to tune the collision radius
    collision_radius = collision_radius_over_spacing * spacing_for_drones

    # note that all the list/array indices are representing the agent id
    loc_x_prev_t_ = global_state_["loc_x"][time_step - 1]
    loc_y_prev_t_ = global_state_["loc_y"][time_step - 1]
    loc_z_prev_t_ = global_state_["loc_z"][time_step - 1]

    # loc_prev = (loc_x_prev_t_, loc_y_prev_t_, loc_z_prev_t_)
    interpolation = 20
    # linear interpolation of agents' paths with constant spacing of 1/2 collision radius
    x_paths = np.array(
        [np.linspace(loc_x_prev_t_[id_], loc_x_curr_t_[id_], interpolation, dtype=float_dtype)]
        for id_ in range(_num_agents_))

    y_paths = np.array(
        [np.linspace(loc_y_prev_t_[id_], loc_y_curr_t_[id_], interpolation, dtype=float_dtype)]
        for id_ in range(_num_agents_))

    z_paths = np.array(
        [np.linspace(loc_z_prev_t_[id_], loc_z_curr_t_[id_], interpolation, dtype=float_dtype)]
        for id_ in range(_num_agents_))

    # check collisions with buildings
    impact_with_buildings = \
        impact_check_first(_num_agents_, city_map_, x_paths, y_paths, z_paths)

    impact_with_buildings, collision_set, collision_with_agents = \
        collision_check_then(
            _num_agents_,
            collision_radius,
            impact_with_buildings,
            x_paths, y_paths, z_paths)

    # resolving impact
    (speed_curr_t_, vert_speed_curr_t_), dir_curr_t, (loc_x_curr_t_, loc_y_curr_t_, loc_z_curr_t_) = \
        resolve_impact(
            num_agents=_num_agents_,
            impact_damping=0.5,
            impact_with_buildings=impact_with_buildings,
            loc_x_curr_t=loc_x_curr_t_,
            loc_y_curr_t=loc_y_curr_t_,
            loc_z_curr_t=loc_z_curr_t_,
            speed_curr_t=speed_curr_t_,
            vert_speed_curr_t=vert_speed_curr_t_,
            dir_curr_t=dir_curr_t_)

    # resolving collision
    speed_curr_t_, vert_speed_curr_t_, dir_curr_t_ = \
        resolve_collision(
            collision_pair_set=collision_set,
            collision_with_agents=collision_with_agents,
            speed_curr_t=speed_curr_t_,
            vert_speed_curr_t=vert_speed_curr_t_,
            dir_curr_t=dir_curr_t)

    # todo: return these agents id for calculating rewards
    impact_agent_ids = [ag_id for ag_id in impact_with_buildings.keys()]
    collision_agent_ids = [ag_id for ag_id in collision_with_agents.keys()]

    return (loc_x_curr_t_, loc_y_curr_t_, loc_z_curr_t_), (speed_curr_t_, vert_speed_curr_t_, dir_curr_t_), \
        impact_agent_ids, collision_agent_ids
