import numpy as np
import heapq
from gym import spaces

from utils.spawning import *
from utils.generating import generate_city
from utils.rendering import *

from warp_drive.utils.constants import Constants

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS

"""These are all agents' locations and infos"""
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_LOC_Z = "loc_z"

_SP = "speed"
_SP_VERT = "vertical speed"
_DIR = "direction"
_ACC = "acceleration"
_ACC_VERT = "vertical acceleration"
_SIG = "still_in_the_game"

"""These are buildings locations"""
_CITY_LOC = "city_loc"

float_dtype = np.float32
int_dtype = np.int32

"""Essential parameters"""

num_drones: int = 10
num_goals: int = 1
num_obstacles: int = 10

episode_length: int = 100
env_size: int = 20
env_max_height: int = 20
env_max_building_height: int = 18
env_difficulty: int = 1
spacing_for_drones: float = 0.5
spawn_goal_mode: str = "central"
spawn_goal_distance = float(0.25 * env_size)
spawn_goal_direction = np.array([0, 1, 0], dtype=int_dtype)
seed = None
base_speed: float = 1.0
max_drone_speed: float = 1.0
max_goal_speed: float = 1.0
max_obstacle_speed: float = 1.0
max_drone_vert_speed: float = 0.5
max_goal_vert_speed: float = 0.5
max_obstacle_vert_speed: float = 0.5
max_acceleration: float = 1.0
min_acceleration: float = -1.0
max_vert_acceleration: float = 0.5
min_vert_acceleration: float = -0.5
max_turn: float = np.pi / 2
min_turn: float = -np.pi / 2
num_acceleration_levels: int = 10
num_turn_levels: int = 10
edge_hit_penalty: float = -0.0
use_full_observation: bool = False
is_goals_directly_observable: bool = False
drone_obs_range: float = 0.02
pursuit_reward_coef_multiplier: float = 0.25
flock_reward_coef_multiplier: float = 0.5
collision_penalty_coef_multiplier: int = 80
cross_penalty_ceof_multiplier: int = 40
time_penalty_coef_multiplier: int = 1
stab_penalty_coef_multiplier: int = 1
flock_threshold: float = 1.5
collision_threshold: float = 0.5
cross_threshold: float = 1.5
pursuit_threshold: float = 0.7
env_backend = "cpu"

"""Initialize agents"""


def spawning_test(
        _num_agents: int,
        _num_drones: int,
        _num_goals: int,
        _num_obstacles: int,
        _drones: np.ndarray,
        _agent_type: dict,
        _spacing_for_drones: float,
        _spawn_goal_distance,
        _spawn_goal_direction,
        _env_size: int,
        _env_max_building_height: int,
        _env_max_height: int,
        _env_difficulty: int,
        _is_goals_directly_observable: bool
        ):
    for agents_id in range(_num_agents - _num_goals):
        if agents_id in set(_drones):
            agent_type[agents_id] = 1  # drones
            d[agents_id] = True
        else:
            agent_type[agents_id] = 1 / 2  # obstacles
            o[agents_id] = True

    for goal_id in range(_num_agents - _num_goals, _num_agents):
        agent_type[goal_id] = 0  # goal
        g[goal_id] = True

    drone_centroid = np.array(
        [_env_size / 2 - 0.5, 0.5, _env_size / 2 - 0.5],
        dtype=float_dtype
    )

    # d_id_list = [
    #     i for i in range(num_agents) if agent_type[i] == 1
    # ]

    _d_dict = spawn_drones_central(
        num_drones=_num_drones,
        num_agents=_num_agents,
        agent_type=_agent_type,
        step_size=_spacing_for_drones,
        drone_centroid=drone_centroid
    )

    _g_dict = spawn_goal_central(
        num_goals=_num_goals,
        num_agents=_num_agents,
        drone_centroid=drone_centroid,
        agent_type=_agent_type,
        step_size=_spacing_for_drones,
        distance=_spawn_goal_distance,
        direction=_spawn_goal_direction
    )

    _o_dict = spawn_obstacles(
        env_size=_env_size,
        max_height=_env_max_height,
        min_height=5,
        place_radius=0.4,
        num_obstacles=_num_obstacles,
        num_agents=_num_agents,
        agent_type=_agent_type,
        d_dict=_d_dict,
        g_dict=_g_dict
    )

    _city_map = generate_city(
        env_size=_env_size,
        max_buildings_height=_env_max_building_height,
        difficulty_level=_env_difficulty,
        is_directly_obs=_is_goals_directly_observable,
        d_centroid=drone_centroid,
        d_dict=_d_dict,
        o_dict=_o_dict,
        g_dict=_g_dict
    )

    return _d_dict, _o_dict, _g_dict, _city_map


def set_global_state(_global_state, key=None, value=None, t=None, dtype=None):
    """
    Set the global state for a specified key, value and timestamp.
    Note: for a new key, initialize the global state to all zeros.
    """
    assert key is not None
    if dtype is None:
        dtype = float_dtype

    # if no values are passed, set everything to zeros
    if key not in global_state:
        _global_state[key] = np.zeros(
            (episode_length + 1, num_agents), dtype=dtype
        )

    if t is not None and value is not None:
        # print(global_state[key].shape)
        assert isinstance(value, np.ndarray)
        assert value.shape[0] == global_state[key].shape[1]

        _global_state[key][t] = value


def reset(global_state_: dict,
          t: int,
          ep_length: int,
          num_agents_: int,
          starting_location_x_,
          starting_location_y_,
          starting_location_z_,
          starting_speeds_,
          starting_vert_speeds_,
          starting_directions_,
          starting_accelerations_,
          starting_vert_accelerations_):
    """
    Env reset().
    """
    # reset time to the beginning

    # re-initialize the global state
    # print(starting_location_x)
    set_global_state(
        global_state_, key=_LOC_X, value=starting_location_x_, t=t
    )
    set_global_state(
        global_state_, key=_LOC_Y, value=starting_location_y_, t=t
    )
    set_global_state(
        global_state_, key=_LOC_Z, value=starting_location_z_, t=t
    )
    set_global_state(global_state_, key=_SP, value=starting_speeds_, t=t)
    set_global_state(global_state_, key=_SP_VERT, value=starting_vert_speeds_, t=t)
    set_global_state(global_state_, key=_DIR, value=starting_directions_, t=t)
    set_global_state(
        global_state_, key=_ACC, value=starting_accelerations_, t=t
    )
    set_global_state(
        global_state_, key=_ACC_VERT, value=starting_vert_accelerations_, t=t
    )
    # we would not pass city location, because it is a dictionary and not updated in step method.
    # set_global_state(key=_CITY_LOC, value=buildings, t=timestep)

    # initialize global state for "still_in_the_game" to all ones
    global_state[_SIG] = np.ones(
        (ep_length + 1, num_agents_), dtype=int_dtype
    )
    # Reinitialize variables that may have changed during previous episode

    # array to keep track of the agents that are still in game
    still_in_the_game = np.ones(num_agents_, dtype=int_dtype)

    return global_state_, still_in_the_game


"""Test version of methods from env class"""

if __name__ == "__main__":
    """These are all agents' locations and infos"""
    _LOC_X = "loc_x"
    _LOC_Y = "loc_y"
    _LOC_Z = "loc_z"

    _SP = "speed"
    _SP_VERT = "vertical speed"
    _DIR = "direction"
    _ACC = "acceleration"
    _ACC_VERT = "vertical acceleration"
    _SIG = "still_in_the_game"

    """These are buildings locations"""
    _CITY_LOC = "city_loc"

    float_dtype = np.float32
    int_dtype = np.int32
    # first initialize the agents
    num_agents = num_goals + num_obstacles + num_drones

    drones = np.random.choice(
        np.arange(num_agents - num_goals), num_drones, replace=False
    )
    agent_type = {}
    d = {}
    g = {}
    o = {}

    global_state = {}
    timestep = 0

    # build the dictionary for each type of agents, key would be the universal agent's id
    """Spawning"""

    d_dict, o_dict, g_dict, city_map = \
        spawning_test(
            _num_agents=num_agents,
            _num_drones=num_drones,
            _num_goals=num_goals,
            _num_obstacles=num_obstacles,
            _drones=drones,
            _agent_type=agent_type,
            _spacing_for_drones=spacing_for_drones,
            _spawn_goal_distance=spawn_goal_distance,
            _spawn_goal_direction=spawn_goal_direction,
            _env_size=env_size,
            _env_max_building_height=env_max_building_height,
            _env_max_height=env_max_height,
            _env_difficulty=env_difficulty,
            _is_goals_directly_observable=is_goals_directly_observable
        )

    """Populating state dictionary"""
    buildings = {}

    for x in range(env_size):
        for y in range(env_size):
            if city_map[x, y]:
                buildings[(x, y)] = city_map[x, y]

    # collect all agents' starting x, y, z locations
    # todo: we make sure the dictionary is sorted by keys here
    # so that we can access coords by using agent id as index in global state
    points_start_dict = dict(sorted({**d_dict, **g_dict, **o_dict}.items()))
    # print(points_start_dict)

    starting_location_x = np.array(
        [coords[0] for coords in points_start_dict.values()]
    )

    starting_location_y = np.array(
        [coords[1] for coords in points_start_dict.values()]
    )

    starting_location_z = np.array(
        [coords[2] for coords in points_start_dict.values()]
    )

    # assign and collect all agents' starting directions
    starting_directions = np.random.choice(
        [0, np.pi / 2, np.pi, np.pi * 3 / 2], num_agents, replace=True
    )

    # set up agents kinematics
    # set the max speed level (planar)
    max_drone_speed = float_dtype(max_drone_speed)
    max_goal_speed = float_dtype(max_goal_speed)
    max_obstacle_speed = float_dtype(max_obstacle_speed)

    # set up max speed level (vertical)
    max_drone_vert_speed = float_dtype(max_drone_vert_speed)
    max_goal_vert_speed = float_dtype(max_goal_vert_speed)
    max_obstacle_vert_speed = float_dtype(max_obstacle_vert_speed)

    # initialize agents acceleration (all with 0) and speeds
    # All agents start with 0 speed and acceleration
    starting_speeds = np.zeros(num_agents, dtype=float_dtype)
    # todo: incorporate vertical speed in state dictionary
    starting_vert_speeds = np.zeros(num_agents, dtype=float_dtype)
    starting_accelerations = np.zeros(num_agents, dtype=float_dtype)
    # todo: incorporate vertical acceleration in state dictionary
    starting_vert_accelerations = np.zeros(num_agents, dtype=float_dtype)

    assert num_acceleration_levels >= 0
    assert num_turn_levels >= 0

    num_acceleration_levels = num_acceleration_levels
    num_turn_levels = num_turn_levels
    max_acceleration = float_dtype(max_acceleration)
    min_acceleration = float_dtype(min_acceleration)
    max_vert_acceleration = float_dtype(max_vert_acceleration)
    min_vert_acceleration = float_dtype(min_vert_acceleration)

    max_turn = float_dtype(max_turn)
    min_turn = float_dtype(min_turn)

    # Acceleration actions (planar)
    acceleration_actions = np.linspace(
        min_acceleration, max_acceleration, num_acceleration_levels
    )

    # Acceleration actions (vertical)
    vert_acceleration_actions = np.linspace(
        min_vert_acceleration, max_vert_acceleration, num_acceleration_levels
    )

    # Add action 0 - this will be the no-op, or 0 acceleration
    acceleration_actions = np.insert(acceleration_actions, 0, 0).astype(
        float_dtype
    )

    # Add action 0 - this will be the no-op, or 0 vert acceleration
    vert_acceleration_actions = np.insert(
        vert_acceleration_actions, 0, 0).astype(float_dtype)

    # Turn actions (only planar)
    turn_actions = np.linspace(
        min_turn, max_turn, num_turn_levels
    )

    # Add action 0 - this will be the no op, or 0 direction
    direction_actions = np.insert(turn_actions, 0, 0).astype(float_dtype)

    # defining observation and action spaces

    # assume that agents can climb/descend, planar move, and turn at the same time.
    action_space = {
        agent_id: spaces.MultiDiscrete(
            [len(acceleration_actions),
             len(vert_acceleration_actions),
             len(turn_actions)]
        )
        for agent_id in range(num_agents)
    }

    """Rewards and penalties"""

    assert pursuit_reward_coef_multiplier > 0
    pursuit_rewards_coef = float_dtype(pursuit_reward_coef_multiplier / env_size)

    assert flock_reward_coef_multiplier > 0
    flock_rewards_coef = float_dtype(flock_reward_coef_multiplier / env_size)

    assert collision_penalty_coef_multiplier > 0
    collision_penalty_coef = float_dtype(collision_penalty_coef_multiplier / env_size)

    assert cross_penalty_ceof_multiplier > 0
    cross_penalty_coef = float_dtype(cross_penalty_ceof_multiplier / env_size)

    assert time_penalty_coef_multiplier > 0
    time_penalty_coef = float_dtype(time_penalty_coef_multiplier / env_size)

    assert stab_penalty_coef_multiplier > 0
    stab_penalty_coef = float_dtype(stab_penalty_coef_multiplier / env_size)

    """Execute and return"""
    global_state, still_in_the_game = \
        reset(
            global_state_=global_state,
            t=timestep,
            ep_length=episode_length,
            num_agents_=num_agents,
            starting_location_x_=starting_location_x,
            starting_location_y_=starting_location_y,
            starting_location_z_=starting_location_z,
            starting_speeds_=starting_speeds,
            starting_vert_speeds_=starting_vert_speeds,
            starting_directions_=starting_directions,
            starting_accelerations_=starting_accelerations,
            starting_vert_accelerations_=starting_vert_accelerations
        )

    print("-- test rigs ready! --")
