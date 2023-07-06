import copy
import heapq
import random

import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext
from utils.spawning import *
from utils.generating import *
from utils.collision import *

# TODO: figure out what these constants are for...
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS

"""These are all agents' locations and infos"""
DRONES = 1
OBSTACLES = 1 / 2
GOALS = 0
_DRONES_IDS = "drones_ids"
_OBS_IDS = "obstacles_ids"
_GOALS_IDS = "goals_ids"
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_LOC_Z = "loc_z"

_SP = "speed"
_V_SP = "vertical speed"
_DIR = "direction"
_ACC = "acceleration"
_V_ACC = "vertical acceleration"
_SIG = "still_in_the_game"

"""These are buildings locations"""
_CITY_LOC = "city_loc"
_B_INFO = "buildings_info"
_B_AABB = "buildings_aabb"


class DroneReachContinuous(CUDAEnvironmentContext):

    def __init__(
            self,
            num_drones: int = 1,
            num_goals: int = 1,
            num_obstacles: int = 10,
            episode_length: int = 100,
            env_size: int = 10,
            env_max_height: int = 20,
            env_max_building_height: int = 18,
            env_difficulty: int = 1,
            spacing_for_drones: float = 0.3,
            spawn_goal_mode: str = "central",
            spawn_goal_distance=None,
            spawn_goal_direction=None,
            seed=None,
            base_speed: float = 1.0,
            max_drone_speed: float = 1.0,
            max_goal_speed: float = 1.0,
            max_obstacle_speed: float = 1.0,
            max_drone_vert_speed: float = 0.5,
            max_goal_vert_speed: float = 0.5,
            max_obstacle_vert_speed: float = 0.5,
            max_acceleration: float = 1.0,
            min_acceleration: float = -1.0,
            max_vert_acceleration: float = 0.5,
            min_vert_acceleration: float = -0.5,
            max_turn: float = np.pi / 2,
            min_turn: float = -np.pi / 2,
            num_acceleration_levels: int = 10,
            num_turn_levels: int = 10,
            edge_hit_penalty: float = -0.0,
            use_full_observation: bool = False,
            use_communication: bool = True,
            is_goals_directly_observable: bool = False,
            drone_sensing_range_multiplier: float = 0.02,
            pursuit_reward_coef_multiplier: float = 0.25,
            flock_reward_coef_multiplier: float = 0.5,
            impact_penalty_coef_multiplier: int = 80,
            collision_penalty_ceof_multiplier: int = 40,
            time_penalty_coef_multiplier: int = 1,
            stab_penalty_coef_multiplier: int = 1,
            flock_threshold: float = 1.5,
            agents_radius: float = 0.1,
            pursuit_threshold: float = 0.7,
            env_backend="cpu",
    ):
        """
        Args:

            num_drones:(int, optional): number of agents in the environment.
                Defaults to 1.
            num_goals:(int, optional): number of goals in the environment.
                Defaults to 1.
            num_obstacles:(int, optional): number of moving obstacles in the environment.
                Defaults to 10.
            episode_length:(int, optional): length of episode.
                Defaults to 100.
            env_size:(int, optional): size of the environment.
                Defaults to 10.
            env_difficulty:(int, optional): difficulty level of the environment, range between 1 and 9 (both exclusive).
                Defaults to 1.
            spacing_for_drones:(float, optional): spacing when spawning the drone.
                Defaults to 0.3.
            spawn_goal_mode:(str, optional): spawning mode for goal.
                Defaults to "central".
            env_max_height:(int, optional): maximum height of the environment that agents and goals can act in.
                Defaults to 20.
            env_max_building_height:(int, optional): maximum height of buildings in the environment.
                Defaults to 18.
            seed:([type], optional): [seeding parameter].
                Defaults to None.
            base_speed:(float, optional): base speed for agents and goals.
                Defaults to 1.
            max_drone_speed:(float, optional): a base speed multiplier for drones.
                Defaults to 1.
            max_goal_speed:(float, optional): a base speed multiplier for goals.
                Defaults to 1.
            max_obstacle_speed:(float, optional): a base speed multiplier for obstacles.
                Defaults to 1.
            max_acceleration:(float, optional): the max acceleration.
                Defaults to 1.0.
            min_acceleration:(float, optional): the min acceleration.
                Defaults to -1.0.
            max_turn: Defaults to np.pi/2.
            min_turn: Defaults to -np.pi/2.
            num_acceleration_levels:(int, optional): number of acceleration actions uniformly spaced between min and max
                acceleration. Defaults to 10.
            num_turn_levels:(int, optional): number of turn actions uniformly spaced
                between max and min turns.
                Defaults to 10.
            edge_hit_penalty:(float, optional): penalty for hitting the edge.
                Defaults t0 -0.0
            use_full_observation:(bool, optional): boolean indicating whether to include all the agents' data in the
                observation or just the ones within observation range. Defaults to False.
            is_goals_directly_observable:(bool, optional): boolean indicating whether the goals are directly observable
                for agents at initial conditions.
                Defaults to False.
            drone_sensing_range_multiplier:(float, optional): range of drones' sensors. This multiplies on top of the grid length.
                Defaults to 0.02.
            pursuit_reward_coef_multiplier: (float, optional): multiplied over 1 / env_size, to yield the
                coefficient for pursuit rewards for individual drones.
                Defaults to 0.25.
            flock_reward_coef_multiplier: (float, optional): multiplied over 1 / env_size, to yield the
                coefficient for flocking rewards for all drones.
                Defaults to 0.5.
            impact_penalty_coef_multiplier: (int, optional): multiplied over 1 / env_size, to yield the
                coefficient for collision penalties for individual drones with obstacles and buildings.
                Defaults to 80.
            collision_penalty_ceof_multiplier: (int, optional): multiplied over 1 / env_size, to yield the
                coefficient for collision / crossing penalties with other drones.
                Defaults to 40.
            time_penalty_coef_multiplier: (int, optional): multiplied over 1 / env_size, to yield the
                coefficient for time penalty for each drone per time-step taken to solve the task.
                This is only meaningful in navigation tasks, which have a finite horizon.
                Defaults to 1.
            stab_penalty_coef_multiplier: (int, optional): multiplied over 1 / env_size, to yield the
                coefficient for stability penalties for each drone.
                Defaults to 1.
            flock_threshold: (float, optional): threshold distance between each drone and flock centroid in
                calculating flocking rewards.
                Defaults to 1.5 units.
            pursuit_threshold: (float, optional): threshold distance between nearest drone(s) to the goal for calculating
                pursuit rewards.
                Defaults to 0.7 units.
            env_backend:(string, optional): indicate whether to use the CPU or the GPU (either pycuda or numba) for
                stepping through the environment. Defaults to "cpu".
            """
        super().__init__()

        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)

        assert num_drones > 0
        self.num_drones = num_drones

        assert num_goals > 0
        self.num_goals = num_goals

        # note: number of obstacles might be changed in the reset
        assert num_obstacles > 0
        self.num_obstacles = num_obstacles

        self.num_agents = self.num_obstacles + self.num_goals + self.num_obstacles

        assert episode_length >= 0
        self.episode_length = episode_length

        assert env_size > 0
        self.env_size = self.int_dtype(env_size)
        self.env_diagonal = self.env_size * np.sqrt(2)

        # note: env_difficulty might be changed in the reset
        assert env_difficulty > 0
        self.difficulty_level = env_difficulty

        assert spacing_for_drones < 1
        self.spacing_for_drones = spacing_for_drones

        # todo: yet to support sparse spawning for goals
        assert spawn_goal_mode is "central"
        self.spawn_goal_mode = spawn_goal_mode

        # todo: yet to support sparse spawning for goals
        if spawn_goal_distance is not None:
            self.spawn_goal_distance = self.float_dtype(spawn_goal_distance * self.env_size)
        else:
            self.spawn_goal_distance = self.float_dtype(0.25 * self.env_size)
            pass

        # todo: the spawn goal direction may not work for multi-goals situations
        if spawn_goal_direction is not None:
            assert isinstance(spawn_goal_direction, np.ndarray)
            assert spawn_goal_direction.shape[0] == 3
            self.spawn_goal_direction = spawn_goal_direction
        else:
            self.spawn_goal_direction = np.array([1, 0, 0], dtype=int_dtype)

        assert env_max_height > 0
        self.env_max_height = env_max_height

        assert env_max_building_height > 0
        assert env_max_building_height < env_max_height
        self.buildings_max_height = env_max_building_height

        # Note that drone_sensing_range_multiplier is a env size multiplier.

        assert edge_hit_penalty <= 0
        self.edge_hit_penalty = self.float_dtype(edge_hit_penalty)

        # seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        # starting drone's ids list, and obstacle's ids list
        self.drones_ids = np.zeros(self.num_drones)
        self.goals_ids = np.zeros(self.num_goals)
        self.obstacles_ids = np.zeros(self.num_obstacles)
        # how about setting these in reset??
        self.agent_type = {}
        self.drones = {}
        self.goals = {}
        self.obstacles = {}
        self.buildings_info = {}
        self.buildings_aabb = {}
        self.starting_location_x = None
        self.starting_location_y = None
        self.starting_location_z = None
        self.starting_directions = None
        # set up agents kinematics
        # set the max speed level (planar)
        self.max_drone_speed = self.float_dtype(max_drone_speed)
        self.max_goal_speed = self.float_dtype(max_goal_speed)
        self.max_obstacle_speed = self.float_dtype(max_obstacle_speed)

        # set up max speed level (vertical)
        self.max_drone_vert_speed = self.float_dtype(max_drone_vert_speed)
        self.max_goal_vert_speed = self.float_dtype(max_goal_vert_speed)
        self.max_obstacle_vert_speed = self.float_dtype(max_obstacle_vert_speed)

        # initialize agents acceleration (all with 0) and speeds
        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_vert_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_vert_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0

        self.num_acceleration_levels = num_acceleration_levels
        self.num_turn_levels = num_turn_levels
        self.max_acceleration = self.float_dtype(max_acceleration)
        self.min_acceleration = self.float_dtype(min_acceleration)
        self.max_vert_acceleration = self.float_dtype(max_vert_acceleration)
        self.min_vert_acceleration = self.float_dtype(min_vert_acceleration)

        self.max_turn = self.float_dtype(max_turn)
        self.min_turn = self.float_dtype(min_turn)

        # Acceleration actions (planar)
        self.acceleration_actions = np.linspace(
            self.min_acceleration, self.max_acceleration, self.num_acceleration_levels
        )

        # Acceleration actions (vertical)
        self.vert_acceleration_actions = np.linspace(
            self.min_vert_acceleration, self.max_vert_acceleration, self.num_acceleration_levels
        )

        # Add action 0 - this will be the no-op, or 0 acceleration
        self.acceleration_actions = np.insert(self.acceleration_actions, 0, 0).astype(
            self.float_dtype
        )

        # Add action 0 - this will be the no-op, or 0 vert acceleration
        self.vert_acceleration_actions = np.insert(
            self.vert_acceleration_actions, 0, 0).astype(self.float_dtype)

        # Turn actions (only planar)
        self.turn_actions = np.linspace(
            self.min_turn, self.max_turn, self.num_turn_levels
        )

        # Add action 0 - this will be the no op, or 0 direction
        self.direction_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)

        # defining observation and action spaces

        # assume that agents can climb/descend, planar move, and turn at the same time.
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                [len(self.acceleration_actions),
                 len(self.vert_acceleration_actions),
                 len(self.turn_actions)]
            )
            for agent_id in range(self.num_agents)
        }

        # Used in generate_observation()
        # When use_full_observation is True, then all agents will have info of
        # all the other agents are environment, otherwise, each agent will only have info of
        # agents and environment within its observation range.
        # Note that sometimes buildings will obstruct local observation.
        # todo: we set up observation and rewards info here

        # always enable agents communications in local observability settings
        assert use_communication == ~use_full_observation
        self.use_full_observation = use_full_observation
        self.use_communication = use_communication
        if self.num_goals > 1:
            self.multi_goals_tracking = True
            self.compute_goal_shortest_path = False
        else:
            self.multi_goals_tracking = False
            self.compute_goal_shortest_path = True
        self.init_obs = None  # will set later in generate_observation()

        assert flock_threshold > 0
        self.flock_threshold = flock_threshold

        assert drone_sensing_range_multiplier > 0
        self.drone_sensing_range = drone_sensing_range_multiplier * self.env_size
        # detection range for goal halved
        self.drone_sensing_range_for_goal = self.drone_sensing_range / 2
        self.neighbor_cut_off_dis = self.drone_sensing_range
        self.comm_cut_off_dis = self.drone_sensing_range
        self.is_directly_observable = is_goals_directly_observable

        # Rewards and penalties
        assert pursuit_reward_coef_multiplier > 0
        self.pursuit_rewards_coef = self.float_dtype(pursuit_reward_coef_multiplier / self.env_size)

        assert flock_reward_coef_multiplier > 0
        self.flock_rewards_coef = self.float_dtype(flock_reward_coef_multiplier / self.env_size)

        assert impact_penalty_coef_multiplier > 0
        self.collision_penalty_coef = self.float_dtype(impact_penalty_coef_multiplier / self.env_size)

        assert collision_penalty_ceof_multiplier > 0
        self.cross_penalty_coef = self.float_dtype(collision_penalty_ceof_multiplier / self.env_size)

        assert agents_radius > 0
        self.agents_radius = agents_radius

        assert time_penalty_coef_multiplier > 0
        self.time_penalty_coef = self.float_dtype(time_penalty_coef_multiplier / self.env_size)

        assert stab_penalty_coef_multiplier > 0
        self.stab_penalty_coef = self.float_dtype(stab_penalty_coef_multiplier / self.env_size)

        # Compiling step/time penalty rewards for all types of agents
        # basically, convert the agent type dictionary into a corresponding list
        # such that given the agent_id as index of the list
        # we can directly retrieve their step rewards/penalty
        # without querying the agent types
        # todo: this will be calculated in compute rewards part
        self.step_rewards = None
        # self.step_rewards = [
        #     self.agent_type[agents_id] * self.step_penalty_for_drone
        #     + (1 - self.agent_type[agents_id]) * self.step_reward_for_goal
        #     for agents_id in range(self.num_agents)
        # ]

        # These would be set during reset
        self.timestep = None
        self.global_state = None

        # todo: this would be set later
        # todo: useful only in multi-goal/evader scenario
        self.still_in_the_game = None

        # These will also be set via the env_wrapper
        self.env_backend = env_backend

        # Copy agents dict for applying at reset
        # todo: what else do we need to copy or change for the reset?
        self.agents_at_reset = copy.deepcopy(self.agent_type)

    name = "DroneReachContinuous"

    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_global_state(self, key=None, value=None, t=None, dtype=None):
        """
        Set the global state for a specified key, value and timestamp.
        Note: for a new key, initialize the global state to all zeros.
        """
        assert key is not None
        if dtype is None:
            dtype = self.float_dtype

        # if no values are passed, set everything to zeros
        if key not in self.global_state:
            self.global_state[key] = np.zeros(
                (self.episode_length + 1, self.num_agents), dtype=dtype
            )

        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == self.global_state[key].shape[1]

            self.global_state[key][t] = value

    def update_state(self, delta_accelerations, delta_vert_accelerations, delta_turns):
        """
        Note: 'update_state' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (update_state)
        is part of the step() function!

        The logic below mirrors (part of) the step function in CUDA.
        """
        # todo: verify that all of the quantities involved are in the format of numpy arrays.
        loc_x_prev_t = self.global_state[_LOC_X][self.timestep - 1]
        loc_y_prev_t = self.global_state[_LOC_Y][self.timestep - 1]
        loc_z_prev_t = self.global_state[_LOC_Z][self.timestep - 1]
        speed_prev_t = self.global_state[_SP][self.timestep - 1]
        dir_prev_t = self.global_state[_DIR][self.timestep - 1]
        acc_prev_t = self.global_state[_ACC][self.timestep - 1]
        vert_speed_prev_t = self.global_state[_V_SP][self.timestep - 1]
        vert_acc_prev_t = self.global_state[_V_ACC][self.timestep - 1]
        bs_info = self.global_state[_B_INFO][0]
        bs_aabb = self.global_state[_B_AABB][0]

        # Update direction and acceleration
        dir_curr_t = (
                (dir_prev_t + delta_turns) % (2 * np.pi)
        ).astype(self.float_dtype)

        acc_curr_t = acc_prev_t + delta_accelerations
        vert_acc_curr_t = vert_acc_prev_t + delta_vert_accelerations
        max_speed = [
            self.max_drone_speed if self.agent_type[i] == 1 else
            self.max_obstacle_speed if self.agent_type[i] == 1 / 2 else
            self.max_goal_speed for i in range(self.num_agents)
        ]
        max_vert_speed = [
            self.max_drone_vert_speed if self.agent_type[i] == 1 else
            self.max_obstacle_vert_speed if self.agent_type[i] == 1 / 2 else
            self.max_goal_vert_speed for i in range(self.num_agents)
        ]
        speed_curr_t = self.float_dtype(
            np.clip(speed_prev_t + acc_curr_t, 0.0, max_speed) * self.still_in_the_game
        )
        vert_speed_curr_t = self.float_dtype(
            np.clip(vert_speed_prev_t + vert_acc_curr_t, 0.0, max_vert_speed) * self.still_in_the_game
        )
        # reset acc and v_acc to 0 if over-speed, fine to reset them here before resolving
        # todo: may need to test this
        acc_curr_t = acc_curr_t * (speed_curr_t > 0) * (speed_curr_t < max_speed)
        vert_acc_curr_t = vert_acc_curr_t * (vert_speed_curr_t > 0) * (vert_speed_curr_t < max_vert_speed)

        # re-arrange prev_locations, curr_speeds for spheres_step
        speed_curr_t_x = speed_curr_t * np.cos(dir_curr_t)
        speed_curr_t_y = speed_curr_t * np.sin(dir_curr_t)
        speed_curr_t_z = vert_speed_curr_t
        temp = np.vstack((speed_curr_t_x, speed_curr_t_y, speed_curr_t_z))
        speed_curr_t_vectors = np.swapaxes(temp, 0, 1)  # -> shape: (num_agents SIG, 3)

        # re-arranging the locations
        temp_loc_prev_t = np.vstack((loc_x_prev_t, loc_y_prev_t, loc_z_prev_t))
        loc_prev_t = np.swapaxes(temp_loc_prev_t, 0, 1)  # -> shape: (num_agents SIG, 3)

        # fed it to step_spheres function
        loc_curr_t, resolved_speed_curr_t, recorders = step_spheres(
            self.env_max_height, self.num_agents, loc_prev_t, speed_curr_t_vectors, bs_info, bs_aabb, self.agents_radius
        )

        # unwrap returned data shape (num_agents SIG, 3) -> (3, num_agents SIG)
        # do not use reshape or transpose here! It will mess up the entries!
        reshaped_loc_curr_t = np.swapaxes(loc_curr_t, 0, 1)
        loc_x_curr_t = reshaped_loc_curr_t[0]
        loc_y_curr_t = reshaped_loc_curr_t[1]
        loc_z_curr_t = reshaped_loc_curr_t[2]

        # retrieved speed and direction components (num_agents SIG, 3) -> (3, num_agents SIG), "r" for resolved
        reshaped_speed_curr_t = np.swapaxes(resolved_speed_curr_t, 0, 1)
        r_vert_speed_curr_t = reshaped_speed_curr_t[2]
        r_speed_x_curr_t = reshaped_speed_curr_t[0]
        r_speed_y_curr_t = reshaped_speed_curr_t[1]

        r_dir_curr_t = np.arctan2(r_speed_y_curr_t, r_speed_x_curr_t, dtype=float_dtype)

        stacked_x_y = np.vstack((r_speed_x_curr_t, r_speed_y_curr_t))
        planar_speed_curr_t = np.swapaxes(stacked_x_y, 0, 1)
        r_speed_curr_t = np.linalg.norm(planar_speed_curr_t, axis=1)

        # todo: the agents that are removed from game are still passing in info!
        # todo: can we find a way to remove them from the info directly?
        # todo: the SIG is updated in compute_rewards part

        # unwrap the recorders. which are all lists
        loc_recorder, speed_recorder, event_recorder = recorders

        # fed them through compute rewards function

        # penalize drones for collisions

        # set global states
        self.set_global_state(key=_LOC_X, value=loc_x_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Y, value=loc_y_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Z, value=loc_z_curr_t, t=self.timestep)
        self.set_global_state(key=_SP, value=r_speed_curr_t, t=self.timestep)
        self.set_global_state(key=_V_SP, value=r_vert_speed_curr_t, t=self.timestep)
        self.set_global_state(key=_DIR, value=r_dir_curr_t, t=self.timestep)
        self.set_global_state(key=_ACC, value=acc_curr_t, t=self.timestep)
        self.set_global_state(key=_V_ACC, value=vert_acc_curr_t, t=self.timestep)

        pass

    def compute_loc_obs(self, loc_x, loc_y, loc_z, speed_x, speed_y, speed_z):
        """
        Compute all drones' local observation of detected buildings, obstacles.

        Notes:
            - The return format are dictionaries, whose keys are agent ids, values being a 2D numpy array of shape
                (num_of_entities_observed, num_observation_quantities).
            - For obstacles speed observation, the quantities are [norm, pitch, yaw].
            - For obstacles distance observation, the quantities are [norm, pitch, yaw]
            - For buildings observation, the quantities are [norm, pitch, yaw, b_dim_x, b_dim_y, b_dim_z].

        Warnings:
            - This function is part of `generate_observation` routine, and only runs for the CPU version of this env.

        """
        # extract info
        drones_loc = np.vstack(
            (loc_x[self.drones_ids], loc_y[self.drones_ids], loc_z[self.drones_ids])
        )  # shape: (num_drones, 3)
        drones_speed = np.vstack(
            (speed_x[self.drones_ids], speed_y[self.drones_ids], speed_z[self.drones_ids])
        )  # shape: (num_drones, 3)
        obstacles_loc = np.vstack(
            (loc_x[self.obstacles_ids], loc_y[self.obstacles_ids], loc_z[self.obstacles_ids])
        )  # shape: (num_obstacles, 3)
        obstacles_speed = np.vstack(
            (loc_x[self.obstacles_ids], loc_y[self.obstacles_ids], loc_z[self.obstacles_ids])
        )
        # compute relative velocities and distances
        local_obstacles_speed_obs = {}
        local_obstacles_dis_obs = {}
        local_buildings_obs = {}
        edges_walk = 0.5 * np.array([[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0]])
        v_edge_unit = np.array([0, 0, 1])
        for drone_index in range(self.num_drones):
            drone_id = self.drones_ids[drone_index]
            d_loc = drones_loc[drone_id]
            o_local_speed_obs = []  # use self.drones_id to translate index back to agent ids, -> np.array
            o_local_dis_obs = []
            b_local_obs = []
            # determine observed obstacles
            for _index in range(self.num_obstacles):
                o_rel_dis_vec = obstacles_loc[_index] - d_loc
                o_rel_dis_norm = np.linalg.norm(o_rel_dis_vec)
                if o_rel_dis_norm[_index] < self.drone_sensing_range or self.use_full_observation:
                    o_rel_dis_planar_norm = np.linalg.norm(o_rel_dis_vec[0:2])
                    o_rel_dis_pitch = np.arctan2(o_rel_dis_vec[2], o_rel_dis_planar_norm)
                    o_rel_dis_yaw = np.arctan2(o_rel_dis_vec[1], o_rel_dis_vec[0])

                    o_rel_speed_vec = obstacles_speed[_index] - drones_speed[drone_index]
                    o_rel_speed_norm = np.linalg.norm(o_rel_speed_vec)
                    o_rel_speed_planar_norm = np.linalg.norm(o_rel_speed_vec[0:2])
                    o_rel_speed_pitch = np.arctan2(o_rel_speed_vec[2], o_rel_speed_planar_norm)
                    o_rel_speed_yaw = np.arctan2(o_rel_speed_vec[1], o_rel_dis_vec[0])

                    o_local_speed_obs.append([o_rel_speed_norm, o_rel_speed_yaw, o_rel_speed_pitch])
                    o_local_dis_obs.append([o_rel_dis_norm, o_rel_dis_yaw, o_rel_dis_pitch])

            local_obstacles_speed_obs[drone_id] = np.array(o_local_speed_obs)
            local_obstacles_dis_obs[drone_id] = np.array(o_local_dis_obs)

            # determine the observed buildings
            # a problem of determining if a sphere intersects a vertical cube
            # use the sphere's aabb rep to check
            # if intersects, verify the closest edge to sphere center <= sphere radius

            s_aabb_centroid = d_loc
            s_aabb_dim = 2 * self.agents_radius * np.array([1, 1, 1])
            s_aabb = [s_aabb_centroid, s_aabb_dim]
            for building in self.buildings_info.keys():
                observed = False  # simple flag for local observability.
                b_centroid = np.array(building)
                b_aabb = self.buildings_aabb[building]
                if not self.use_full_observation:
                    # compute the closest vertical edges
                    b_top_centroid = np.array([b_centroid[0], b_centroid[1], 2 * b_centroid[2]])
                    b_bottom_centroid = array([b_centroid[0], b_centroid[1], 0])
                    v_edges_end = b_top_centroid + edges_walk
                    v_edges_start = b_bottom_centroid + edges_walk
                    v_edge_norm = 2 * b_centroid[2]
                    # use relevant geometry functions to do the job
                    if check_aabb_intersection(b_aabb, s_aabb):
                        # find the closest vertical edge
                        planar_edge_dis = np.zeros(4)
                        edge_idx = 0
                        for v_edge in v_edges_start:
                            planar_dis = np.linalg.norm(d_loc[0:2] - v_edge[0:2])
                            planar_edge_dis[edge_idx] = planar_dis
                        edge_idx = np.argmin(planar_edge_dis)
                        # check the sphere center-edge segment distance
                        closest_edge_dis = helper_distance_point_to_line_segment(
                            d_loc, v_edges_start[edge_idx], v_edges_end[edge_idx], v_edge_norm, v_edge_unit
                        )
                        if closest_edge_dis <= self.drone_sensing_range:
                            observed = True
                if self.use_full_observation or observed:
                    # currently we only return the relative building centroid vector, and its dimensions
                    b_rel_dis_vec = b_centroid - d_loc
                    b_rel_dis_norm = np.linalg.norm(b_rel_dis_vec)
                    b_rel_dis_planar_norm = np.linalg.norm(b_rel_dis_vec[0:2])
                    b_rel_pitch = np.arctan2(b_rel_dis_vec[2], b_rel_dis_planar_norm)
                    b_rel_yaw = np.arctan2(b_rel_dis_vec[1], b_rel_dis_vec[0])
                    _, b_dimension = b_aabb
                    b_x, b_y, b_z = b_dimension
                    b_local_obs.append([b_rel_dis_norm, b_rel_pitch, b_rel_yaw, b_x, b_y, b_z])

            local_buildings_obs[drone_id] = np.array(b_local_obs)

        return local_obstacles_speed_obs, local_obstacles_dis_obs, local_buildings_obs

    def compute_drones_obs(self, loc_x, loc_y, loc_z, speed_x, speed_y, speed_z):
        """
        Compute all drones' local observation of neighboring drones, and the graph representation of neighborhood. The
        local observations are returned as dictionaries. The neighborhood graph info includes a graph edge list,
        and a local neighbor dictionary.
        Notes:
            - Return orders are `speed`, `distances`, `graph edges`, and `local neighbors`.
            - Local observation values of shape `(num_entities_observed, num_observed_quantities)`.
            - For neighboring drones speed, quantities are `[norm, yaw, pitch]`.
            - For neighboring drones distances, quantities are `[norm, yaw, pitch]`.
            - For neighbors,  graph_edge_list with pairs of drone_id within the `neighbor_cut_off_dis`.
            - For local neighbors, local_neighbors dict with a list of drone_id within **local** `neighbor_cut_off_dis`

        """
        drones_loc = np.vstack(
            (loc_x[self.drones_ids], loc_y[self.drones_ids], loc_z[self.drones_ids])
        )  # shape: (num_drones, 3)
        drones_speed = np.vstack(
            (speed_x[self.drones_ids], speed_y[self.drones_ids], speed_z[self.drones_ids])
        )  # shape: (num_drones, 3)

        neighbor_idx_matrix = np.zeros((self.num_agents, self.num_agents))
        graph_edge_list = []
        local_neighbors = {}  # key being agent ids, holds a list of ag_id that are neighbors (sub-graph)
        local_drones_speed_obs = {}
        local_drones_dis_obs = {}

        for drone_index in range(self.num_drones):
            drone_id = self.drones_ids[drone_index]
            d_loc = drones_loc[drone_id]
            d_local_neighbors = []
            d_local_speed_obs = []  # use self.drones_id to translate index back to agent ids, -> np.array
            d_local_dis_obs = []  # use self.drones_id to translate index back to agent ids, -> np.array

            # determine drones neighborhood
            other_index_list = list(range(0, drone_index)) + list(range(drone_index + 1, self.num_drones))
            for other_drone_idx in other_index_list:
                other_drone_id = self.drones_ids[other_drone_idx]
                # in Numba, calculate relevant quantities here without using axis and tile
                rel_dis_vec = drones_loc[other_drone_idx] - d_loc
                rel_dis_norm = np.linalg.norm(rel_dis_vec)
                if rel_dis_norm < self.drone_sensing_range or self.use_full_observation:
                    rel_dis_planar_norm = np.linalg.norm(rel_dis_vec[0:2])
                    rel_dis_pitch_angle = np.arctan2(rel_dis_vec[2], rel_dis_planar_norm)
                    rel_dis_yaw_angle = np.arctan2(rel_dis_vec[1], rel_dis_vec[0])

                    rel_speed_vec = drones_speed[other_drone_idx] - drones_speed[drone_index]
                    rel_speed_norm = np.linalg.norm(rel_speed_vec)
                    rel_speed_planar_norm = np.linalg.norm(rel_speed_vec[0:2])
                    rel_speed_pitch_angle = np.arctan2(rel_speed_vec[2], rel_speed_planar_norm)
                    rel_speed_yaw_angle = np.arctan2(rel_speed_vec[1], rel_dis_vec[0])

                    d_local_speed_obs.append([rel_speed_norm, rel_speed_yaw_angle, rel_speed_pitch_angle])
                    d_local_dis_obs.append([rel_dis_norm, rel_dis_yaw_angle, rel_dis_pitch_angle])

                    if rel_dis_norm < self.neighbor_cut_off_dis:
                        d_local_neighbors.append(other_drone_id)
                        neighbor_idx_matrix[drone_index][other_drone_idx] = 1
                        neighbor_idx_matrix[other_drone_idx][drone_index] = 1
                        edge = {drone_id, other_drone_id}
                        if edge not in graph_edge_list:
                            graph_edge_list.append(edge)

                local_drones_speed_obs[drone_id] = np.array(d_local_speed_obs)
                local_drones_dis_obs[drone_id] = np.array(d_local_dis_obs)
                local_neighbors[drone_id] = d_local_neighbors

        return (local_drones_speed_obs, local_drones_dis_obs), graph_edge_list, local_neighbors

    def compute_goals_obs(self, loc_x, loc_y, loc_z, speed_x, speed_y, speed_z):
        """
        Compute the local observation of goals. The return formats are dictionaries indicating each
        agent's local observations of the goals. If the goal is observed, relative speeds and location.
        Returns:

        """
        drones_loc = np.vstack(
            (loc_x[self.drones_ids], loc_y[self.drones_ids], loc_z[self.drones_ids])
        )  # shape: (num_drones, 3)
        drones_speed = np.vstack(
            (speed_x[self.drones_ids], speed_y[self.drones_ids], speed_z[self.drones_ids])
        )  # shape: (num_drones, 3)
        goals_loc = np.vstack(
            (loc_x[self.goals_ids], loc_y[self.goals_ids], loc_z[self.goals_ids])
        )
        goals_speed = np.vstack(
            (speed_x[self.goals_ids], speed_y[self.goals_ids], speed_z[self.goals_ids])
        )
        local_goals_dis_obs = {}
        local_goals_speed_obs = {}
        local_goals_neighbor = {}
        drones_goals_edges = []
        for drone_index in range(self.num_drones):
            drone_id = self.drones_ids[drone_index]
            d_loc = drones_loc[drone_index]
            g_local_speed_obs = []
            g_local_dis_obs = []
            g_local_neighbor = []
            for g_index in range(self.num_goals):
                g_loc = goals_loc[g_index]
                goal_id = self.goals_ids[g_index]
                rel_dis_vec = g_loc - d_loc
                rel_dis_norm = np.linalg.norm(rel_dis_vec)
                if rel_dis_norm < self.drone_sensing_range_for_goal or self.use_full_observation:
                    rel_dis_planar_norm = np.linalg.norm(rel_dis_vec[0:2])
                    rel_dis_pitch_angle = np.arctan2(rel_dis_vec[2], rel_dis_planar_norm)
                    rel_dis_yaw_angle = np.arctan2(rel_dis_vec[1], rel_dis_vec[0])

                    rel_speed_vec = goals_speed[g_index] - drones_speed[drone_index]
                    rel_speed_norm = np.linalg.norm(rel_speed_vec)
                    rel_speed_planar_norm = np.linalg.norm(rel_speed_vec[0:2])
                    rel_speed_pitch_angle = np.arctan2(rel_speed_vec[2], rel_speed_planar_norm)
                    rel_speed_yaw_angle = np.arctan2(rel_speed_vec[1], rel_dis_vec[0])

                    g_local_speed_obs.append([rel_speed_norm, rel_speed_yaw_angle, rel_speed_pitch_angle])
                    g_local_dis_obs.append([rel_dis_norm, rel_dis_yaw_angle, rel_dis_pitch_angle])

                    # or: if rel_dis_norm < self.drone_sensing_range / 2:
                    if rel_dis_norm < self.drone_sensing_range_for_goal:
                        drones_goals_edges.append([drone_id, goal_id])
                        g_local_neighbor.append(goal_id)
                    else:
                        # todo: we use -1 for None value if the goal is not in agent's neighborhood.
                        g_local_neighbor.append(-1)

            local_goals_speed_obs[drone_id] = np.array(g_local_speed_obs)
            local_goals_dis_obs[drone_id] = np.array(g_local_dis_obs)
            local_goals_neighbor[drone_id] = g_local_neighbor

        return local_goals_speed_obs, local_goals_dis_obs, local_goals_neighbor

    #     # todo: where do we check for the "still_in_the_game"?
    #     # todo: we would simply re-spawn the goal agents after they have been captured
    def compute_shortest_path(self, start, target, adjacency_matrix):
        max_iter = self.num_agents  # number of edges no more than total number of agents
        num_nodes = len(adjacency_matrix)
        distances = [float('inf')] * num_nodes
        distances[start] = 0
        queue = [(0, start)]
        parent = {}
        iter_num = 0
        while queue and iter_num < max_iter:
            current_distance, current_node = heapq.heappop(queue)
            if current_node == target:
                path = [current_node]
                while current_node != start:
                    current_node = parent[current_node]
                    path.append(current_node)
                return path[::-1], distances

            if current_distance > distances[current_node]:
                continue

            for neighbor in range(num_nodes):
                weight = adjacency_matrix[current_node][neighbor]
                if weight > 0:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(queue, (distance, neighbor))
                        parent[neighbor] = current_node
            iter_num += 1
        return None

    def compute_graph_property(self, loc_x, loc_y, loc_z, graph_edges, drones_goals_edges):
        """
        Utilize the information in neighborhood and connected edges between goal-drone and drones, compute additional
        information that might be useful in communications in local observability settings.
        Notes:
            - In single goal tracking, shortest path distance from local agents to goal is computed.
            - In multi-gaols tracking, shortest path distances from local agents to each goal are computed.
            - If no such values are found, use a default value of -1 instead.
        Args:
            loc_x: x coordinates of all agents.
            loc_y: y coordinates of all agents.
            loc_z: z coordinates of all agents.
            drones_goals_edges: a list of lists that holds a pair of nodes (agent id) indicating an edge.
            graph_edges: a list of sets that each holds a pair of nodes (agent id) indicating an edge.

        Returns:
            a dictionary whose keys are drone's agent ids, values are a list that holds the shortest path distances of
            the drone to each of the goal. If no such path is found, a default value of -1 is supplemented for that goal.

        """
        loc = np.vstack((loc_x, loc_y, loc_z))  # shape: (num_agents, 3)
        # construct the adjacency matrix with weights being the dis
        total_num = self.num_goals + self.num_drones
        index_to_id = np.concatenate((self.drones_ids, self.goals_ids))
        id_to_index = {index_to_id[i]: i for i in range(total_num)}
        adjacency_matrx = np.full((total_num, total_num), -1)  # default value is -1
        # first input the entries for drones
        all_edges = graph_edges + drones_goals_edges
        for pair in all_edges:
            agent_id1, agent_id2 = pair
            d_loc1 = loc[agent_id1]
            d_loc2 = loc[agent_id2]
            weight = np.linalg.norm(d_loc1 - d_loc2)
            index1 = id_to_index[agent_id1]
            index2 = id_to_index[agent_id2]
            adjacency_matrx[index1, index2] = weight
            adjacency_matrx[index2, index1] = weight

        # then for each drone, and each goal, check the shortest paths
        drone_goal_shortest_dis = {}
        for drone_id in self.drones_ids:
            drone_index = id_to_index[drone_id]
            goals_shortest_dis = []
            for goal_id in self.goals_ids:
                goal_index = id_to_index[goal_id]
                res = self.compute_shortest_path(drone_index, goal_index, adjacency_matrx)
                if res is not None:
                    _, distances = res
                    shortest_dis = distances[goal_index]
                    goals_shortest_dis.append(shortest_dis)
                else:
                    goals_shortest_dis.append(-1)  # use -1 for default values of not seeing the goals

            drone_goal_shortest_dis[drone_id] = goals_shortest_dis

        return drone_goal_shortest_dis

    def generate_observation(self):
        """
        Generate and return the observations for every agent
        Each agent receives three components which form their local observations, these are:\n
        - local features: observed buildings, borders, obstacles, and possible, goals information.\n
        - neighbors info: neighboring agents' relative positions, and relative velocities.\n
        - neighbors comm: neighboring agents' communication message by the comm protocols. \n

        For global observability, agents have access to all the information, and every agent is connected.

        For local observability, agents only have access to local features within its observation cut off distances,
        and neighbors info within its comm cut off distance, and neighbors comm from neighbors within its comm cut off
        distances.
        """
        obs = {}
        if self.use_full_observation is False:

            pass
        else:
            pass
        return obs

    def compute_reward(self,
                       prev_speed_vector,
                       curr_speed_vector,
                       loc_recorder,
                       speed_recorder,
                       event_recorder):
        """
        Compute and return the rewards for each agent.
        """
        # cast inputs to numpy arrays
        if not isinstance(prev_speed_vector, np.ndarray):  # shape: (num_agents, 3)
            prev_speed_vector = np.array(prev_speed_vector)
        if not isinstance(curr_speed_vector, np.ndarray):  # shape: (num_agents, 3)
            curr_speed_vector = np.array(curr_speed_vector)
        if not isinstance(event_recorder, np.ndarray):  # shape: (_fps, num_agents)
            event_recorder = np.array(event_recorder)
        if not isinstance(speed_recorder, np.ndarray):  # shape: (_fps, num_agents, 3)
            speed_recorder = np.array(speed_recorder)
        if not isinstance(loc_recorder, np.ndarray):  # shape: (_fps, num_agents, 3)
            loc_recorder = np.array(loc_recorder)

        # initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        # we will calculate the rewards based on each time-step instead of the start-end state
        # so that our rewards signal are much smoother
        # todo: attenuate the rewards coefficient by fps (No. of sub-steps in each environment stepping)
        # flocking rewards
        # we need to extract the drones coordinates here

        drones_list = sorted(self.drones)

        # todo: goal reward for single-goal and multi-goals are different
        if self.num_goals > 0:
            goals_list = sorted(self.goals)
            goals_locations_x = self.global_state[_LOC_X][self.timestep][goals_list]
            drones_locations_x = self.global_state[_LOC_X][self.timestep][drones_list]

            goals_locations_y = self.global_state[_LOC_Y][self.timestep][goals_list]
            drones_locations_y = self.global_state[_LOC_Y][self.timestep][drones_list]

            goals_locations_z = self.global_state[_LOC_Z][self.timestep][goals_list]
            drones_locations_z = self.global_state[_LOC_Z][self.timestep][drones_list]

            # todo: what the heck this following repeat and tile are doing?
            drones_to_goals_distances = np.sqrt(
                (
                        np.tile(goals_locations_x, self.num_drones)
                        - np.repeat(drones_locations_x, self.num_goals)
                )
                ** 2
                + (
                        np.tile(goals_locations_y, self.num_drones)
                        - np.repeat(drones_locations_y, self.num_goals)
                )
                ** 2
                + (
                        np.tile(goals_locations_z, self.num_drones)
                        - np.repeat(drones_locations_z, self.num_goals)
                )
                ** 2
            ).reshape(self.num_goals, self.num_drones)

            # min_drones_to_goals_distances = np.min(
            #     drones_to_goals_distances, axis=1
            # )
            #
            # argmin_drones_to_goals_distances = np.argmin(
            #     drones_to_goals_distances, axis=1
            # )
            #
            # nearest_drones_ids = [
            #     drones_list[idx] for idx in argmin_drones_to_goals_distances
            # ]

        # Rewards

        return rew

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _LOC_Z, _SP, _DIR, _ACC, _CITY_LOC]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )
        data_dict.add_data(
            name="agent_types",
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
            save_copy_and_apply_at_reset=True
        )
        data_dict.add_data(
            name="num_drones", data=self.num_drones
        )

        pass

    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        return tensor_dict

    def reset(self):
        """
        Env reset().
        """
        # reset time to the beginning
        self.timestep = 0
        self.drones_ids = self.np_random.choice(
            np.arange(self.num_agents), self.num_drones, replace=False
        )
        # spawn agents and generate here instead!!!
        # build the dictionary for each type of agents, key would be the universal agent's id
        obstacles_idx = 0
        goals_idx = 0
        for agents_id in range(self.num_agents - self.num_goals):
            if agents_id in set(self.drones_ids):
                self.agent_type[agents_id] = DRONES
                self.drones[agents_id] = True
            else:
                self.agent_type[agents_id] = OBSTACLES
                self.obstacles[agents_id] = True
                self.obstacles_ids[obstacles_idx] = agents_id
                obstacles_idx += 1

        # append goal information at last
        for goal_id in range(self.num_agents - self.num_goals, self.num_agents):
            self.agent_type[goal_id] = GOALS
            self.goals[goal_id] = True
            self.goals_ids[goals_idx] = goal_id
            goals_idx += 1

        # todo: how about spawning drones at the center of the map?
        drone_centroid = np.array(
            [self.env_size / 2 - 0.5, 0.5, self.env_size / 2 - 0.5],
            dtype=self.float_dtype
        )
        d_dict = spawn_drones_central(
            num_drones=self.num_drones,
            num_agents=self.num_agents,
            agent_type=self.agent_type,
            step_size=self.spacing_for_drones,
            drone_centroid=drone_centroid
        )

        # todo: what if multi-goals?
        if self.spawn_goal_mode is "central":
            g_dict = spawn_goal_central(
                num_goals=self.num_goals,
                num_agents=self.num_agents,
                drone_centroid=drone_centroid,
                agent_type=self.agent_type,
                step_size=self.spacing_for_drones,
                distance=self.spawn_goal_distance,
                direction=self.spawn_goal_direction
            )
        else:  # todo: this is just a placeholder as of now
            g_dict = spawn_goal_central(
                num_goals=self.num_goals,
                num_agents=self.num_agents,
                drone_centroid=drone_centroid,
                agent_type=self.agent_type,
                step_size=self.spacing_for_drones,
                distance=self.spawn_goal_distance,
                direction=self.spawn_goal_direction
            )

        o_dict = spawn_obstacles(
            env_size=self.env_size,
            max_height=self.env_max_height,
            min_height=5,
            place_radius=0.4,
            num_obstacles=self.num_obstacles,
            num_agents=self.num_agents,
            agent_type=self.agent_type,
            d_dict=d_dict,
            g_dict=g_dict
        )

        city_map = generate_city(
            env_size=self.env_size,
            max_buildings_height=self.buildings_max_height,
            difficulty_level=self.difficulty_level,
            is_directly_obs=self.is_directly_observable,
            d_centroid=drone_centroid,
            d_dict=d_dict,
            o_dict=o_dict,
            g_dict=g_dict
        )

        # todo: generate the city buildings' aabb info
        self.buildings_info, self.buildings_aabb = compute_city_info(self.env_max_height, city_map)

        # collect all agents' starting x, y, z locations
        points_start_dict = {**d_dict, **g_dict, **o_dict}

        self.starting_location_x = np.array(
            coords[0] for coords in points_start_dict.values()
        )

        self.starting_location_y = np.array(
            coords[1] for coords in points_start_dict.values()
        )

        self.starting_location_z = np.array(
            coords[2] for coords in points_start_dict.values()
        )

        # assign and collect all agents' starting directions
        self.starting_directions = self.np_random.choice(
            [0, np.pi / 2, np.pi, np.pi * 3 / 2], self.num_agents, replace=True
        )

        # re-initialize the global state
        self.global_state = {}
        self.set_global_state(
            key=_DRONES_IDS, value=self.drones_ids, t=self.timestep
        )
        self.set_global_state(
            key=_OBS_IDS, value=self.obstacles_ids, t=self.timestep
        )
        self.set_global_state(
            key=_GOALS_IDS, value=self.goals_ids, t=self.timestep
        )
        self.set_global_state(
            key=_LOC_X, value=self.starting_location_x, t=self.timestep
        )
        self.set_global_state(
            key=_LOC_Y, value=self.starting_location_y, t=self.timestep
        )
        self.set_global_state(
            key=_LOC_Z, value=self.starting_location_z, t=self.timestep
        )
        self.set_global_state(key=_SP, value=self.starting_speeds, t=self.timestep)
        self.set_global_state(key=_V_SP, value=self.starting_vert_speeds, t=self.timestep)
        self.set_global_state(key=_DIR, value=self.starting_directions, t=self.timestep)
        self.set_global_state(
            key=_ACC, value=self.starting_accelerations, t=self.timestep
        )
        self.set_global_state(
            key=_V_ACC, value=self.starting_vert_accelerations, t=self.timestep
        )
        self.set_global_state(
            key=_B_INFO, value=self.buildings_info, t=self.timestep
        )
        self.set_global_state(
            key=_B_AABB, value=self.buildings_aabb, t=self.timestep
        )

        # array to keep track of the agents that are still in game
        self.still_in_the_game = np.ones(self.num_agents, dtype=self.int_dtype)

        # initialize global state for "still_in_the_game" to all ones
        self.global_state[_SIG] = np.ones(
            (self.episode_length + 1, self.num_agents), dtype=self.int_dtype
        )
        # Reinitialize variables that may have changed during previous episode

        return self.generate_observation()

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        if not self.env_backend == "cpu":
            # CUDA version of step
            # This subsumes update_state(), generate_observation(),
            # and compute_reward()
            pass
        else:
            assert isinstance(actions, dict)
            assert len(actions) == self.num_agents

        acceleration_action_ids = [
            actions[agent_id][0] for agent_id in range(self.num_agents)
        ]
        turn_action_ids = [
            actions[agent_id][1] for agent_id in range(self.num_agents)
        ]

        assert all(
            0 <= acc <= self.num_acceleration_levels
            for acc in acceleration_action_ids
        )
        assert all(0 <= turn <= self.num_turn_levels for turn in turn_action_ids)

        delta_accelerations = self.acceleration_actions[acceleration_action_ids]
        delta_turns = self.turn_actions[turn_action_ids]

        # Update state and generate observation
        self.update_state(delta_accelerations, delta_turns)
        if self.env_backend == "cpu":
            obs = self.generate_observation()

        # Compute rewards and done
        # rew = self.compute_reward()
        # done = {
        #     "__all__": (self.timestep >= self.episode_length)
        #                or (self.num_runners == 0)
        # }
        # info = {}
        #
        # result = obs, rew, done, info
        # return result
        pass
