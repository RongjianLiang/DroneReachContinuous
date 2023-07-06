import math
import numpy as np

from utils.miscellaneous import p_dict_to_block_list

"""data type declaration"""
int_dtype = np.int32
float_dtype = np.float32

"""ugly variable declaration"""

even_spawn_rule = np.array(
    [
        [-1, 0, 0], [1, 0, 0],
        [0, 0, -1], [0, 0, 1],
        [-1, 0, 1], [1, 0, -1],
        [1, 0, 1], [-1, 0, -1]
    ], dtype=int_dtype
)

stagger = np.array(
    [0.5, 0.5, 0.5], dtype=float_dtype
)

spawn_cycle = even_spawn_rule.shape[0]


def spawn_goal_sparse(
        num_goals: int,
        num_agents: int,
        drone_centroid: np.ndarray,
        agent_type: dict,
        displacement: float,
        distance: dict,
        direction: dict,
        mode: str = "sparse"):
    pass


def spawn_goal_central(
        num_goals: int,
        num_agents: int,
        drone_centroid: np.ndarray,
        agent_type: dict,
        step_size: float,
        distance: float,
        direction: np.ndarray,
        mode: str = "central"):
    """
    Spawn the goal location, given drone formation.
    \n
    If multiple goals are required to be spawned, spawn them in the
    similar fashion like drones, i.e., spawn around the centroid of the block.
    \n
    Note that with current implementation of line.of.sight, it is strongly
    advisable to have all goals stay inside a single block. If goals occupy
    multiple blocks, the line of sight calculated would not be accurate.
    \n
    And although we could implement a **listed** line of sight, in *central* mode,
    this might bring significant changes to terrain.
    \n
    We might implement another spawning logic similar to obstacles in another mode.

    *Assume 0 is for goal in agent type dictionary*.
    :param num_goals: number of goal agents to spawn
    :param num_agents: number of agents in total
    :param drone_centroid: a numpy array indicating the centroid of the drone.
    :param agent_type: a dictionary whose keys are agents ids, and values being
        the integer representing their types.
    :param step_size: step_size: step size for each step in the spawning rule.
        Suggested value is 0.3.
    :param distance: a floating point number indicating distance between drone
        block centroid and goals centroid.
    :param direction: a numpy array indicating direction vector from drone
        block centroid to goals centroid.
        Suggested value is np.array([0, 1, 0], dtype=int_dtype)
    :param mode: mode of spawning. This function default to "central".
        Spawn the goal according to the spawn rule around a given centroid.
    :return: a dictionary whose keys are goal agents ids,
        and values being their coordinates.
    """
    assert isinstance(num_agents, int)
    assert isinstance(num_goals, int)
    assert num_goals >= 1
    assert num_agents > num_goals
    assert isinstance(drone_centroid, np.ndarray) and drone_centroid.shape[0] == 3
    assert isinstance(agent_type, dict)
    assert isinstance(distance, float)
    assert isinstance(step_size, float)
    assert isinstance(direction, np.ndarray) and direction.shape[0] == 3
    assert mode is "central"

    distance = float_dtype(distance)
    step_size = float_dtype(step_size)
    num_goals = int_dtype(num_goals)
    num_agents = int_dtype(num_agents)

    g_id_list = [
        i for i in range(num_agents) if agent_type[i] == 0
    ]

    assert len(g_id_list) == num_goals

    g_dict = {}

    g_centroid = drone_centroid + distance * direction
    print(g_centroid)

    if num_goals == 1:  # naive case
        g_dict[g_id_list[0]] = float_dtype(g_centroid)
        return g_dict
    elif num_goals % 2:  # odd number of goals
        g_dict[g_id_list[0]] = float_dtype(g_centroid)
        for g_id in range(num_goals - 1):
            g_dict[g_id_list[g_id + 1]] = float_dtype(
                g_centroid +
                even_spawn_rule[g_id % spawn_cycle] * step_size *
                math.ceil((g_id + 1) / spawn_cycle)
            )
    else:  # even number of goals
        for g_id in range(num_goals):
            g_dict[g_id_list[g_id]] = float_dtype(
                g_centroid +
                even_spawn_rule[g_id % spawn_cycle] * step_size *
                math.ceil((g_id + 1) / spawn_cycle)
            )

    g_blk = p_dict_to_block_list(g_dict)

    assert len(g_blk) == 1
    # all goals should lie inside one block. At least for now.

    return g_dict


def spawn_drones_central(
        num_drones: int,
        num_agents: int,
        agent_type: dict,
        step_size: float,
        drone_centroid: np.ndarray):
    """
    Spawn drones location around a given centroid, with the step size
    for spawning rule.
    *Assume 1 is for drone in agent type dictionary*
    :param num_drones: number of drones to spawn.
    :param num_agents: total number of agents.
    :param agent_type: a dictionary whose keys are agents ids, and values being
        the integer representing their types.
    :param step_size: step size for each step in the spawning rule.
        Suggested value is 0.3.
    :param drone_centroid: a float numpy array for centroid coordinates.
    :return:a dictionary with keys being drone agent ids,
        and values being their coordinates.
    """
    assert step_size > 0.25  # for rendering reasons
    assert num_drones > 1
    assert isinstance(agent_type, dict)
    assert isinstance(num_drones, int)
    assert isinstance(num_agents, int)
    assert isinstance(step_size, float)
    assert isinstance(drone_centroid, np.ndarray) and drone_centroid.shape[0] == 3

    num_drones = int_dtype(num_drones)
    num_agents = int_dtype(num_agents)
    step_size = float_dtype(step_size)

    d_id_list = [
        i for i in range(num_agents) if agent_type[i] == 1
    ]

    assert len(d_id_list) == num_drones

    d_dict = {}

    if num_drones == 1:  # naive case
        d_dict[d_id_list[0]] = float_dtype(drone_centroid)
        return d_dict
    elif num_drones % 2:  # odd number of drones
        d_dict[d_id_list[0]] = float_dtype(drone_centroid)
        for d_id in range(num_drones - 1):
            d_dict[d_id_list[d_id + 1]] = float_dtype(
                drone_centroid +
                even_spawn_rule[d_id % spawn_cycle] * step_size *
                math.ceil((d_id + 1) / spawn_cycle)
            )
    else:  # even number of drones
        for d_id in range(num_drones):
            d_dict[d_id_list[d_id]] = float_dtype(
                drone_centroid +
                even_spawn_rule[d_id % spawn_cycle] * step_size *
                math.ceil((d_id + 1) / spawn_cycle)
            )

    return d_dict


def spawn_obstacles(
        env_size: int,
        max_height: int,
        min_height: int,
        place_radius: float,
        num_obstacles: int,
        num_agents: int,
        agent_type: dict,
        d_dict: dict,
        g_dict: dict):
    """
    Spawn a given number of obstacles in empty blocks, not interfering with
    goals and drones.
    *Assume 0.5 is for obstacles in agent type dictionary*
    :param env_size: size of the environment
    :param max_height: maximum height for spawning. It is suggested to have
        max_height = max_env_height.
    :param min_height: minimum height for spawning. It is suggested to have
        min_height = max_env_height / 2.
    :param place_radius: a displacement distance
        from the center of spawned blocks
    :param num_obstacles: number of obstacles to spawn
    :param num_agents: total number of agents
    :param agent_type: a dictionary whose keys are agents ids, and values being
        the integer representing their types.
    :param d_dict: a dictionary with keys being drone agent ids,
        and values being their coordinates.
    :param g_dict: a dictionary with keys being goal agent ids,
        and values being their coordinates.
    :return: a dictionary with keys being obstacles agent ids,
        and values being their coordinates.
    """
    assert env_size > 0 and isinstance(env_size, int)
    assert max_height > 0 and isinstance(max_height, int)
    assert min_height > 0 and isinstance(min_height, int)
    assert max_height > min_height
    assert 1 > place_radius > 0 and isinstance(place_radius, float)
    assert isinstance(num_agents, int)
    assert isinstance(num_obstacles, int)
    assert isinstance(d_dict, dict)
    assert isinstance(g_dict, dict)

    env_size = int_dtype(env_size)
    max_height = int_dtype(max_height)
    min_height = int_dtype(min_height)
    num_agents = int_dtype(num_agents)
    num_obstacles = int_dtype(num_obstacles)
    place_radius = float_dtype(place_radius)

    o_id_list = [
        i for i in range(num_agents) if agent_type[i] == 0.5
    ]

    assert len(o_id_list) == num_obstacles

    g_blk = p_dict_to_block_list(g_dict)
    d_blk = p_dict_to_block_list(d_dict)

    occ_blk = np.concatenate((g_blk, d_blk))

    empty_blk = np.array(
        [
            np.array([x, y, z]).astype(int_dtype)
            for x in range(env_size)
            for y in range(env_size)
            for z in range(min_height, max_height) if not
        np.any(
            np.all(
                np.array([x, y, z]).astype(int_dtype) == occ_blk,
                axis=1
            )
        )
        ]
    )

    np.random.shuffle(empty_blk)

    o_dict = {
        o_id_list[o_id]:
            np.array(
                empty_blk[o_id] + stagger + np.random.rand(3) * place_radius,
                dtype=float_dtype
            )
        for o_id in range(num_obstacles)
    }

    return o_dict
