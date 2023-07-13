import copy
import heapq

import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext
from utils.spawning import *
from utils.generating import *
from utils.collision import *

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS

"""These are all agents' locations and infos"""
DRONES = 1
OBSTACLES = 1 / 2
GOALS = 0
IMPACT_EVENT = 1
COLLISION_EVENT = 2

# these are the usual keys
# shape for each t: (num_agents, )
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_LOC_Z = "loc_z"

_SP = "speed"
_V_SP = "vertical speed"
_DIR = "direction"
_ACC = "acceleration"
_V_ACC = "vertical acceleration"
_SIG = "still_in_the_game"

"""These are for high FPS rendering"""
# shape for each t: (num_agents, num_recording_per_t)
_R_LOC_X = "Recorder loc_x"
_R_LOC_Y = "Recorder loc_y"
_R_LOC_Z = "Recorder loc_z"
_R_SP_X_ = "Recorder speed_x"
_R_SP_Y_ = "Recorder speed_y"
_R_SP_Z_ = "Recorder speed_z"
_R_EVENT = "Recorder event"


class DroneReachContinuous(CUDAEnvironmentContext):

    def __init__(
            self,
            num_drones: int = 10,
            num_goals: int = 1,
            num_obstacles: int = 10,
            episode_length: int = 200,
            env_size: int = 22,
            env_max_height: int = 20,
            env_max_building_height: int = 18,
            env_difficulty: int = 1,
            env_use_periodic_borders: bool = True,
            spacing_for_drones: float = 0.3,
            seed=None,
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
            use_full_observation: bool = False,
            use_communication: bool = True,
            capture_reward_mode: str = "sparse",
            capture_reward_sparse: float = 1,
            capture_reward_coef: float = 1,
            separation_penalty_coef: float = 40,
            alignment_reward_coef: float = 40,
            coherence_reward_coef: float = 40,
            impact_penalty_coef: float = 0.5,
            collision_penalty_ceof: float = 0.5,
            step_penalty_reward: float = 1,
            stab_penalty_coef_multiplier: int = 1,
            drone_sensing_range: float = 3,
            goal_sensing_range: float = 3,
            drone_sensing_range_for_goals_coef: float = 0.8,
            drone_sensing_range_for_obstacles_coef: float = 0.75,
            drone_sensing_range_for_buildings_coef: float = 0.75,
            neighbor_cut_off_dis: float = 1.5,
            min_separation: float = 0.3,
            min_centroid_drift: float = 0.2,
            agents_radius: float = 0.1,
            capture_threshold: float = 0.2,
            env_backend="cpu",
            record_fps=30,
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
            env_max_height:(int, optional): maximum height of the environment that agents and goals can act in.
                Defaults to 20.
            env_max_building_height:(int, optional): maximum height of buildings in the environment.
                Defaults to 18.
            seed:([type], optional): [seeding parameter].
                Defaults to None.
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
            use_full_observation:(bool, optional): boolean indicating whether to include all the agents' data in the
                observation or just the ones within observation range. Defaults to False.
            drone_sensing_range:(float, optional): range of drones' sensors.
                Defaults to 0.02.
            capture_reward_sparse: (float, optional): multiplied over 1 / env_size, to yield the
                coefficient for pursuit rewards for individual drones.
                Defaults to 0.25.
            step_penalty_reward: (int, optional): multiplied over 1 / env_size, to yield the
                coefficient for time penalty for each drone per time-step taken to solve the task.
                This is only meaningful in navigation tasks, which have a finite horizon.
                Defaults to 1.
            stab_penalty_coef_multiplier: (int, optional): multiplied over 1 / env_size, to yield the
                coefficient for stability penalties for each drone.
                Defaults to 1.
            neighbor_cut_off_dis: (float, optional): threshold distance between each drone and flock centroid in
                calculating flocking rewards.
                Defaults to 1.5 units.
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

        self.num_buildings = None

        self.num_agents = self.num_obstacles + self.num_goals + self.num_obstacles

        assert episode_length >= 0
        self.episode_length = episode_length

        assert env_size > 0
        self.env_size = self.int_dtype(env_size)

        # note: env_difficulty might be changed in the reset
        assert env_difficulty > 0
        self.env_difficulty = env_difficulty
        self.env_use_periodic_borders = env_use_periodic_borders
        assert spacing_for_drones < 1
        assert spacing_for_drones > agents_radius
        self.spacing_for_drones = spacing_for_drones

        assert env_max_height > 0
        self.env_max_height = env_max_height

        assert env_max_building_height > 0
        assert env_max_building_height < env_max_height
        self.env_max_building_height = env_max_building_height

        # seeding
        # this also set up the seeding for all subsequent PRNG
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
        # this would be computed at reset
        self.max_speed = None
        # set up max speed level (vertical)
        self.max_drone_vert_speed = self.float_dtype(max_drone_vert_speed)
        self.max_goal_vert_speed = self.float_dtype(max_goal_vert_speed)
        self.max_obstacle_vert_speed = self.float_dtype(max_obstacle_vert_speed)
        # this will be computed at reset
        self.max_vert_speed = None
        # initialize agents acceleration (all with 0) and speeds
        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_vert_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_vert_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0

        # the acceleration level is for both vert and planar
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
        self._acc_interval = (self.max_acceleration - self.min_acceleration) / self.num_acceleration_levels
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
        self._turn_interval = (self.max_turn - self.min_turn) / self.num_turn_levels
        # Add action 0 - this will be the no op, or 0 direction
        self.turn_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)

        # defining observation and action spaces

        # assume that agents can climb/descend, planar move, and turn at the same time.
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                [len(self.acceleration_actions),
                 len(self.vert_acceleration_actions),
                 len(self.turn_actions)]
            )
            for agent_id in range(self.num_drones)
        }
        # the following is defined for convenience, only used for sampling goals and obstacles action.
        # todo: for Numba implementation, we may not have access to gym.spaces.MultiDiscrete...
        # so just define this _action_space_array with NumPy features...
        # but the random sampling need to have a random seed
        # we would init _action_space_array as a NumPy array here and use NumPy random sample methods for CPU testing
        # but for Numba implementation, check out Numba GPU multi-threaded random function here:
        # https://numba.pydata.org/numba-doc/latest/cuda/random.html
        self._action_space = spaces.MultiDiscrete(
            [len(self.acceleration_actions),
             len(self.vert_acceleration_actions),
             len(self.turn_actions)]
        )
        self._action_space_array = np.array(
            (
                len(self.acceleration_actions),
                len(self.vert_acceleration_actions),
                len(self.turn_actions)
            ), dtype=self.int_dtype
        )

        # always enable agents communications in local observability settings
        assert use_communication == ~use_full_observation
        self.use_full_observation = use_full_observation
        self.use_communication = use_communication
        # if self.num_goals > 1:
        #     self.multi_goals_tracking = True
        #     self.compute_goal_shortest_path = False
        # else:
        #     self.multi_goals_tracking = False
        #     self.compute_goal_shortest_path = True
        self.init_obs = None  # will set later in generate_observation()

        assert neighbor_cut_off_dis > 0
        assert drone_sensing_range > 0
        assert drone_sensing_range_for_buildings_coef < 1
        assert drone_sensing_range_for_obstacles_coef < 1
        assert drone_sensing_range_for_goals_coef < 1
        assert goal_sensing_range > 0
        assert min_separation > 0
        assert neighbor_cut_off_dis < drone_sensing_range
        assert min_separation < neighbor_cut_off_dis
        assert min_centroid_drift < neighbor_cut_off_dis

        self.drone_sensing_range = drone_sensing_range
        # detection range for goal discounted
        self.drone_sensing_range_for_goals = self.drone_sensing_range * drone_sensing_range_for_goals_coef
        # however, the goal can sense the drone at further ranges.
        self.goal_sensing_range = goal_sensing_range
        # drone sensing range for other non-cooperative objects are discounted as well.
        self.drone_sensing_range_for_obstacles = self.drone_sensing_range * drone_sensing_range_for_obstacles_coef
        self.drone_sensing_range_for_buildings = self.drone_sensing_range * drone_sensing_range_for_buildings_coef
        assert capture_threshold < self.drone_sensing_range_for_goals
        self.capture_threshold = capture_threshold
        # we use the concept of neighborhood for flocking
        self.neighbor_cut_off_dis = neighbor_cut_off_dis
        self.min_separation = min_separation
        self.min_centroid_drift = min_centroid_drift
        # initialize the local neighbor dict
        # this would be set at reset
        self.agents_weighted_adj_matrix = None

        # Rewards and penalties
        assert capture_reward_sparse > 0
        self.capture_reward_sparse = self.float_dtype(capture_reward_sparse)
        assert capture_reward_coef > 0
        self.capture_reward_coef = self.float_dtype(capture_reward_coef)
        assert capture_reward_mode is "sparse" or "dense"
        self.capture_reward_mode = capture_reward_mode

        assert separation_penalty_coef > 0
        self.separation_penalty_coef = self.float_dtype(separation_penalty_coef)

        assert alignment_reward_coef > 0
        self.alignment_reward_coef = self.float_dtype(alignment_reward_coef)

        assert capture_reward_coef > 0
        self.coherence_reward_coef = self.float_dtype(coherence_reward_coef)

        assert impact_penalty_coef > 0
        self.impact_penalty_coef = self.float_dtype(impact_penalty_coef)

        assert collision_penalty_ceof > 0
        self.collision_penalty_coef = self.float_dtype(collision_penalty_ceof)

        assert agents_radius > 0
        self.agents_radius = agents_radius

        # Currently however,we are not implementing step-penalty for the drones
        assert step_penalty_reward > 0
        self.step_penalty_reward = self.float_dtype(step_penalty_reward)

        assert stab_penalty_coef_multiplier > 0
        self.stab_penalty_coef = self.float_dtype(stab_penalty_coef_multiplier / self.env_size)

        # Currently however, no step reward or penalty is implemented
        # this will be calculated in compute rewards part
        self.step_rewards = None
        self.step_rewards = [
            self.agent_type[agents_id] * self.step_penalty_reward
            + (1 - self.agent_type[agents_id]) * self.step_penalty_reward
            for agents_id in range(self.num_agents)
        ]

        # These would be set during reset
        self.timestep = None
        self.global_state = None

        self.still_in_the_game = None

        self.record_fps = record_fps
        # These will also be set via the env_wrapper
        self.env_backend = env_backend

        # Copy agents dict for applying at reset
        # todo: what else do we need to copy or change for the reset?
        self.agents_at_reset = copy.deepcopy(self.agent_type)

    name = "DroneReachContinuous"

    def linalg_norm(self, vector: np.ndarray):
        """
        An alias that casts return type to numpy.float32 precision.
        """
        return self.float_dtype(np.linalg.norm(vector))

    def arctan2(self, x1, x2):
        """
        An alias of np.arctan2(x1, x2) that returns numpy.float32 type.
        """
        return self.float_dtype(np.arctan2(x1, x2))

    def arctan(self, x):
        """
        An alias of np.arctan(x) that returns numpy.float32 type.
        """
        return self.float_dtype(np.arctan(x))

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
        # initialize the keys and values for entire episode
        if key not in self.global_state:
            # special shape for recordings
            if key[0] == "R":
                self.global_state[key] = np.zeros(
                    (self.episode_length, self.num_agents, self.record_fps), dtype=float_dtype
                )
            else:
                self.global_state[key] = np.zeros(
                    (self.episode_length + 1, self.num_agents), dtype=dtype
                )
        # then input the init values provided and also subsequent values
        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            # this is to verify that the value input has entries about all agents
            assert value.shape[0] == self.global_state[key].shape[1]
            if key[0] == "R":
                # move 1 time-step ahead for the recorder
                # as they have None for time-step 0
                self.global_state[key][t - 1] = value
            else:
                self.global_state[key][t] = value

    def get_global_speeds(self, agent_id):
        """
        A quick way for retrieving (v_x, v_y, v_z) of given agent in current time-step.
        """
        speed_x = self.global_state[_LOC_X][self.timestep][agent_id]
        speed_y = self.global_state[_LOC_Y][self.timestep][agent_id]
        speed_z = self.global_state[_LOC_Z][self.timestep][agent_id]

        return np.array([speed_x, speed_y, speed_z], dtype=self.float_dtype)

    def get_global_loc(self, agent_id):
        """
        A quick way for retrieving (x, y, z) coordinates of given agent in current time-step.
        """
        loc_x = self.global_state[_LOC_X][self.timestep][agent_id]
        loc_y = self.global_state[_LOC_Y][self.timestep][agent_id]
        loc_z = self.global_state[_LOC_Z][self.timestep][agent_id]

        return np.array([loc_x, loc_y, loc_z], dtype=self.float_dtype)

    def update_state(self, delta_accelerations, delta_vert_acc, delta_turns):
        """
        Note: 'update_state' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (update_state)
        is part of the step() function!

        The logic below mirrors (part of) the step function in CUDA.

        Returns:
            observation dict, reward dict, and possibly info.
        """
        loc_x_prev_t = self.global_state[_LOC_X][self.timestep - 1]
        loc_y_prev_t = self.global_state[_LOC_Y][self.timestep - 1]
        loc_z_prev_t = self.global_state[_LOC_Z][self.timestep - 1]
        speed_prev_t = self.global_state[_SP][self.timestep - 1]
        dir_prev_t = self.global_state[_DIR][self.timestep - 1]
        acc_prev_t = self.global_state[_ACC][self.timestep - 1]
        vert_speed_prev_t = self.global_state[_V_SP][self.timestep - 1]
        vert_acc_prev_t = self.global_state[_V_ACC][self.timestep - 1]

        # Update direction and acceleration
        # use modulus to keep the dir values within bounds
        dir_curr_t = (
                (dir_prev_t + delta_turns) % (2 * np.pi)
        ).astype(self.float_dtype)

        acc_curr_t = acc_prev_t + delta_accelerations
        vert_acc_curr_t = vert_acc_prev_t + delta_vert_acc

        speed_curr_t = self.float_dtype(
            np.clip(speed_prev_t + acc_curr_t, 0.0, self.max_speed) * self.still_in_the_game
        )
        vert_speed_curr_t = self.float_dtype(
            np.clip(vert_speed_prev_t + vert_acc_curr_t, 0.0, self.max_vert_speed) * self.still_in_the_game
        )

        # reset acc and v_acc to 0 if over-speed, fine to reset them here before resolving
        acc_curr_t = acc_curr_t * (speed_curr_t > 0) * (speed_curr_t < self.max_speed)
        vert_acc_curr_t = vert_acc_curr_t * (vert_speed_curr_t > 0) * (vert_speed_curr_t < self.max_vert_speed)

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
            self.env_max_height, self.num_agents, loc_prev_t, speed_curr_t_vectors,
            self.buildings_info, self.buildings_aabb, self.agents_radius
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

        loc_recorder, speed_recorder, event_recorder = recorders
        # cast them to array anyway
        loc_recorder = np.array(loc_recorder)
        speed_recorder = np.array(speed_recorder)
        event_recorder = np.array(event_recorder)

        # proper reshaping
        loc_recorder = np.swapaxes(loc_recorder, 0, 1)  # (num_agents, fps, 3)
        speed_recorder = np.swapaxes(speed_recorder, 0, 1)  # (num_agents, fps, 3)
        event_recorder = np.swapaxes(event_recorder, 0, 1)  # (num_agents, fps)

        # extracting each record for state keys
        loc_x_record = loc_recorder[:, :, 0]
        loc_y_record = loc_recorder[:, :, 1]
        loc_z_record = loc_recorder[:, :, 2]
        speed_x_record = speed_recorder[:, :, 0]
        speed_y_record = speed_recorder[:, :, 1]
        speed_z_record = speed_recorder[:, :, 2]

        # input recorders into global state
        # for recorders and rendering needs
        self.set_global_state(key=_R_LOC_X, value=loc_x_record, t=self.timestep)
        self.set_global_state(key=_R_LOC_Y, value=loc_y_record, t=self.timestep)
        self.set_global_state(key=_R_LOC_Z, value=loc_z_record, t=self.timestep)
        self.set_global_state(key=_R_SP_X_, value=speed_x_record, t=self.timestep)
        self.set_global_state(key=_R_SP_Y_, value=speed_y_record, t=self.timestep)
        self.set_global_state(key=_R_SP_Z_, value=speed_z_record, t=self.timestep)
        self.set_global_state(key=_R_EVENT, value=event_recorder, t=self.timestep)

        # set global states
        self.set_global_state(key=_LOC_X, value=loc_x_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Y, value=loc_y_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Z, value=loc_z_curr_t, t=self.timestep)
        self.set_global_state(key=_SP, value=r_speed_curr_t, t=self.timestep)
        self.set_global_state(key=_V_SP, value=r_vert_speed_curr_t, t=self.timestep)
        self.set_global_state(key=_DIR, value=r_dir_curr_t, t=self.timestep)
        self.set_global_state(key=_ACC, value=acc_curr_t, t=self.timestep)
        self.set_global_state(key=_V_ACC, value=vert_acc_curr_t, t=self.timestep)

        # compute observation and info
        # reset adjacency matrix
        self.agents_weighted_adj_matrix = np.full((self.num_agents, self.num_agents), -1, dtype=self.float_dtype)
        observation = self.generate_observation()
        reward = self.compute_reward()
        # update SIG after determining if any goal is captured
        self.set_global_state(key=_SIG, value=self.still_in_the_game, t=self.timestep)

        info = self.generate_info()

        return observation, reward, info

    def compute_loc_obs(self):
        """
        Compute all drones' local observation of detected buildings, obstacles.

        Returns:
            two dicts with keys being agent ids, values being their observation about obstacles
            and buildings.

        Notes:
            - The return format are dictionaries, whose keys are agent ids, values being a 2D numpy array of shape
                (num_of_entities_observed, num_observation_quantities).
            - For obstacles speed observation, the quantities are [norm, pitch, yaw].
            - For obstacles distance observation, the quantities are [norm, pitch, yaw]
            - For buildings observation, the quantities are [norm, pitch, yaw, b_dim_x, b_dim_y, b_dim_z].

        Warnings:
            - This function is part of `generate_observation` routine, and only runs for the CPU version of this env.

        """
        # compute relative velocities and distances
        local_own_state = {}
        local_obstacles_obs = {}
        local_buildings_obs = {}
        edges_walk = self.float_dtype(0.5 * np.array([[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0]]))
        v_edge_unit = self.int_dtype(np.array([0, 0, 1]))
        # we would compute local neighbour size here as well
        for drone_id in self.drones_ids:
            d_loc = self.get_global_loc(drone_id)
            d_v = self.get_global_speeds(drone_id)
            o_local_speed_obs = []  # use self.drones_id to translate index back to agent ids, -> np.array
            o_local_dis_obs = []
            b_local_obs = []

            # first determine own state
            d_v_planar = self.linalg_norm(d_v[0:2])
            d_pitch = self.arctan2([drone_id], d_v_planar)
            d_yaw = self.arctan2(d_v[1], d_v[0])
            d_v_norm = self.linalg_norm(d_v)
            d_own_state = np.array([d_v_norm, d_pitch, d_yaw])

            # determine neighbourhood size if use communication
            if self.use_communication:
                d_neighbourhood_size = 0
                for o_d_id in self.drones_ids:
                    if o_d_id != drone_id:
                        o_d_loc = self.get_global_loc(o_d_id)
                        rel_d_vec = o_d_loc - d_loc
                        rel_d_norm = self.linalg_norm(rel_d_vec)
                        if rel_d_norm < self.neighbor_cut_off_dis:
                            d_neighbourhood_size += 1
                d_own_state = np.array([d_v_norm, d_pitch, d_yaw, d_neighbourhood_size])

            local_own_state[drone_id] = d_own_state
            # determine observed obstacles
            # we update the adjacency matrix for obstacles as well
            for o_id in self.obstacles_ids:
                o_loc = self.get_global_loc(o_id)
                o_v = self.get_global_speeds(o_id)
                o_rel_dis_vec = o_loc - d_loc
                o_rel_dis_norm = self.linalg_norm(o_rel_dis_vec)
                if o_rel_dis_norm < self.drone_sensing_range_for_obstacles or self.use_full_observation:
                    o_rel_dis_planar_norm = self.linalg_norm(o_rel_dis_vec[0:2])
                    o_rel_dis_pitch = self.arctan2(o_rel_dis_vec[2], o_rel_dis_planar_norm)
                    o_rel_dis_yaw = self.arctan2(o_rel_dis_vec[1], o_rel_dis_vec[0])

                    o_rel_speed_vec = o_v - d_v
                    o_rel_speed_norm = self.linalg_norm(o_rel_speed_vec)
                    o_rel_speed_planar_norm = self.linalg_norm(o_rel_speed_vec[0:2])
                    o_rel_speed_pitch = self.arctan2(o_rel_speed_vec[2], o_rel_speed_planar_norm)
                    o_rel_speed_yaw = self.arctan2(o_rel_speed_vec[1], o_rel_dis_vec[0])

                    o_local_speed_obs.append([o_rel_speed_norm, o_rel_speed_yaw, o_rel_speed_pitch])
                    o_local_dis_obs.append([o_rel_dis_norm, o_rel_dis_yaw, o_rel_dis_pitch])
                    self.agents_weighted_adj_matrix[drone_id, o_id] = o_rel_dis_norm
                    self.agents_weighted_adj_matrix[o_id, drone_id] = o_rel_dis_norm

            # concatenate obstacles observation, speed + distances

            _speed_obs = np.array(o_local_speed_obs)
            _dis_obs = np.array(o_local_dis_obs)
            _obstacles_obs = np.concatenate((_speed_obs, _dis_obs), axis=1)
            local_obstacles_obs[drone_id] = _obstacles_obs
            # determine the observed buildings
            # a problem of determining if a sphere intersects a vertical cube
            # use the sphere's aabb rep to check
            # if intersects, verify the closest edge to sphere center <= sphere radius

            s_aabb_centroid = d_loc
            s_sensing_aabb_dim = 2 * self.drone_sensing_range_for_buildings * np.array([1, 1, 1])
            s_sensing_aabb = [s_aabb_centroid, s_sensing_aabb_dim]
            for building in self.buildings_info.keys():
                observed = False  # simple flag for local observability.
                b_centroid = np.array(building)
                b_aabb = self.buildings_aabb[building]
                if not self.use_full_observation:
                    # compute the closest vertical edges
                    b_top_centroid = np.array([b_centroid[0], b_centroid[1], 2 * b_centroid[2]])
                    b_bottom_centroid = np.array([b_centroid[0], b_centroid[1], 0])
                    v_edges_end = b_top_centroid + edges_walk
                    v_edges_start = b_bottom_centroid + edges_walk
                    v_edge_norm = 2 * b_centroid[2]
                    # use relevant geometry functions to do the job
                    if check_aabb_intersection(b_aabb, s_sensing_aabb):
                        # find the closest vertical edge
                        planar_edge_dis = np.zeros(4)
                        edge_idx = 0
                        for v_edge in v_edges_start:
                            planar_dis = self.linalg_norm(d_loc[0:2] - v_edge[0:2])
                            planar_edge_dis[edge_idx] = planar_dis
                        edge_idx = np.argmin(planar_edge_dis)
                        # check the sphere center-edge segment distance
                        closest_edge_dis = helper_distance_point_to_line_segment(
                            d_loc, v_edges_start[edge_idx], v_edges_end[edge_idx], v_edge_norm, v_edge_unit
                        )
                        if closest_edge_dis <= self.drone_sensing_range_for_buildings:
                            observed = True
                if self.use_full_observation or observed:
                    # currently we only return the relative building centroid vector, and its dimensions
                    b_rel_dis_vec = b_centroid - d_loc
                    b_rel_dis_norm = self.linalg_norm(b_rel_dis_vec)
                    b_rel_dis_planar_norm = self.linalg_norm(b_rel_dis_vec[0:2])
                    b_rel_pitch = self.arctan2(b_rel_dis_vec[2], b_rel_dis_planar_norm)
                    b_rel_yaw = self.arctan2(b_rel_dis_vec[1], b_rel_dis_vec[0])
                    _, b_dimension = b_aabb
                    b_x, b_y, b_z = b_dimension
                    b_local_obs.append([b_rel_dis_norm, b_rel_pitch, b_rel_yaw, b_x, b_y, b_z])

            local_buildings_obs[drone_id] = np.array(b_local_obs)

        return local_own_state, local_obstacles_obs, local_buildings_obs

    def compute_drones_obs(self, local_own_state):
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
        local_drones_obs = {}

        for drone_id in self.drones_ids:
            d_loc = self.get_global_loc(drone_id)
            o_d_v_obs = []
            o_d_d_obs = []
            d_v = self.get_global_speeds(drone_id)
            # determine drones neighborhood
            # we would incorporate local neighbour size here
            # access drone's own state
            for o_d_id in self.drones_ids:
                if o_d_id != drone_id:
                    o_d_loc = self.get_global_loc(o_d_id)
                    o_d_v = self.get_global_speeds(o_d_id)
                    rel_d_vec = o_d_loc - d_loc
                    rel_d_norm = self.linalg_norm(rel_d_vec)
                    if rel_d_norm < self.drone_sensing_range or self.use_full_observation:
                        rel_d_planar_norm = self.linalg_norm(rel_d_vec[0:2])
                        rel_d_pitch = self.arctan2(rel_d_vec[2], rel_d_planar_norm)
                        rel_d_yaw = self.arctan2(rel_d_vec[1], rel_d_vec[0])

                        rel_v = o_d_v - d_v
                        rel_v_norm = self.linalg_norm(rel_v)
                        rel_v_planar_norm = self.linalg_norm(rel_v[0:2])
                        rel_v_pitch = self.arctan2(rel_v[2], rel_v_planar_norm)
                        rel_v_yaw = self.arctan2(rel_v[1], rel_d_vec[0])

                        o_d_v_obs.append([rel_v_norm, rel_v_yaw, rel_v_pitch])
                        o_d_d_obs.append([rel_d_norm, rel_d_yaw, rel_d_pitch])

                        # always keep this adjacency matrix
                        # now we directly update the adjacency matrix with distances,
                        # no need to return relevant information
                        self.agents_weighted_adj_matrix[drone_id, o_d_id] = rel_d_norm
                        self.agents_weighted_adj_matrix[o_d_id, drone_id] = rel_d_norm

                        if self.use_communication:
                            # retrieve their neighbourhood size
                            o_d_neighbourhood_size = local_own_state[o_d_id][-1]
                            o_d_d_obs.append([rel_d_norm, rel_d_yaw, rel_d_pitch, o_d_neighbourhood_size])

                # concatenate speed and distance observation
                d_local_obs = np.concatenate((o_d_v_obs, o_d_d_obs), axis=1)
                local_drones_obs[drone_id] = d_local_obs

        return local_drones_obs

    def compute_goals_obs(self):
        """
        Compute local goal observation and update drone-goal connectivity in adjacency matrix
        when communication is enabled.
        """

        local_goals_obs = {}
        for drone_id in self.drones_ids:
            g_local_speed_obs = []
            g_local_dis_obs = []
            d_loc = self.get_global_loc(drone_id)
            d_v = self.get_global_speeds(drone_id)
            for goal_id in self.goals_ids:
                g_loc = self.get_global_loc(goal_id)
                g_v = self.get_global_speeds(goal_id)
                rel_dis_vec = g_loc - d_loc
                rel_dis_norm = self.linalg_norm(g_loc - d_loc)
                if rel_dis_norm < self.drone_sensing_range_for_goals or self.use_full_observation:
                    rel_dis_planar_norm = self.linalg_norm(rel_dis_vec[0:2])
                    rel_dis_pitch_angle = self.arctan2(rel_dis_vec[2], rel_dis_planar_norm)
                    rel_dis_yaw_angle = self.arctan2(rel_dis_vec[1], rel_dis_vec[0])

                    rel_v = g_v - d_v
                    rel_v_norm = self.linalg_norm(rel_v)
                    rel_v_planar_norm = self.linalg_norm(rel_v[0:2])
                    rel_v_pitch_angle = self.arctan2(rel_v[2], rel_v_planar_norm)
                    rel_v_yaw_angle = self.arctan2(rel_v[1], rel_v[0])

                    # update adjacency matrix
                    self.agents_weighted_adj_matrix[drone_id, goal_id] = rel_dis_norm
                    self.agents_weighted_adj_matrix[goal_id, drone_id] = rel_dis_norm

                    g_local_speed_obs.append([rel_v_norm, rel_v_yaw_angle, rel_v_pitch_angle])
                    g_local_dis_obs.append([rel_dis_norm, rel_dis_yaw_angle, rel_dis_pitch_angle])

                else:
                    # no goals re observed, fill in default values
                    default_v = self.float_dtype(np.array([-1, 2 * np.pi, 2 * np.pi]))
                    default_d = self.float_dtype(np.array([-1, 2 * np.pi, 2 * np.pi]))
                    g_local_speed_obs.append(default_v)
                    g_local_dis_obs.append(default_d)

            g_v_obs = np.array(g_local_speed_obs)
            g_d_obs = np.array(g_local_dis_obs)
            g_obs = np.concatenate((g_v_obs, g_d_obs), axis=1)
            local_goals_obs[drone_id] = g_obs

        return local_goals_obs

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

    def compute_graph_property(self):
        """
        Utilize the information in neighborhood and connected edges between goal-drone and drones, compute additional
        information that might be useful in communications in local observability settings.
        Notes:
            - In single goal tracking, shortest path distance from local agents to goal is computed.
            - In multi-gaols tracking, shortest path distances from local agents to each goal are computed.
            - If no such values are found, use a default value of -1 instead.

        Returns:
            a dictionary whose keys are drone's agent ids, values are a list that holds the shortest path distances of
            the drone to each of the goal.
            If no such path is found, a default value of -1 is supplemented for that goal.
            This list is always of constant length `num_goals`.

        """
        # in the case that drones_goals_edges is empty, just append default values
        drone_goal_shortest_dis = {}
        for drone_id in self.drones_ids:
            goals_shortest_dis = []
            for goal_id in self.goals_ids:
                res = self.compute_shortest_path(drone_id, goal_id, self.agents_weighted_adj_matrix)
                if res is not None:
                    _, distances = res
                    shortest_dis = distances[goal_id]
                    goals_shortest_dis.append(shortest_dis)
                else:
                    goals_shortest_dis.append(-1)  # use -1 for default values of not seeing the goals
            drone_goal_shortest_dis[drone_id] = goals_shortest_dis

        return drone_goal_shortest_dis

    def generate_observation(self):
        """
        Generate and return the observations for every agent.

        Notes:
            Each agent receives three components which form their local observations, these are:
                - local features: observed buildings, borders, obstacles.
                - neighbors info: neighboring agents' relative positions, and relative velocities.
                - goal info: the observation related to goals.
            Observations are computed based on specified flags.
                - local: if global, all buildings, borders, and obstacles are observed.
                - neighbors: if global, all drones are observed.
                - however, only drones within neighborhood cut-off distance are counted as connected.
                - goal: if global, all goals are observed, and the relative distances, bearing are computed.
                - if not, however, only the shortest paths will be computed using neighborhood information.
        """
        local_own_state = local_obstacles_obs, local_buildings_obs = \
            self.compute_loc_obs()
        local_drone_obs, graph_edges = \
            self.compute_drones_obs(local_own_state)
        local_goal_obs = self.compute_goals_obs()

        # self.agents_weighted_adj_matrix = local_neighbors

        if self.use_communication:
            drone_goal_shortest_dis = self.compute_graph_property()
        else:
            drone_goal_shortest_dis = None
        # arrange the local observation into desired formats
        observation = {}
        for drone_id in self.drones_ids:
            _obs = {}
            # the following might vary in size...given local observability
            obstacles_obs = local_obstacles_obs[drone_id]
            building_obs = local_buildings_obs[drone_id]
            other_drones_obs = local_drone_obs[drone_id]

            # goal info will have constant size...
            goal_obs = local_goal_obs[drone_id]

            # this would not be modified later, just put them in first
            _obs["obstacles"] = obstacles_obs
            _obs["buildings"] = building_obs

            # for local observability, append the shortest dis to goal obs as well
            if not self.use_full_observation:
                # update the goal obs with the shortest path to goal
                # the reshaping is for appending,
                # where obs and shortest path about each goal is aligned.
                assert drone_goal_shortest_dis is not None
                shortest_path_length = drone_goal_shortest_dis[drone_id]
                _length = np.array(shortest_path_length).reshape((-1, 1))
                assert _length.shape[0] == goal_obs.shape[0]
                goal_obs = np.append(goal_obs, _length, axis=1)

            _obs["own"] = local_own_state[drone_id]
            _obs["observed_drones"] = other_drones_obs
            _obs["goals"] = goal_obs
            # put the dict inside the overall dict
            observation[drone_id] = _obs

        return observation

    def generate_info(self):
        # compute how many times goals have been captured so far, by accessing the SIG global state
        goals_SIG = self.global_state[_SIG][:, self.goals_ids]
        goals_captured_so_far = np.count_nonzero(goals_SIG == 0)
        steps_per_goals_captured = goals_captured_so_far / self.timestep

        # compute how connected it is for drones
        drone_adjacency_list = self.agents_weighted_adj_matrix[self.drones_ids, self.drones_ids]
        num_drone_connection = np.count_nonzero(drone_adjacency_list > 0)
        num_drone_neighbours = np.count_nonzero(drone_adjacency_list < self.neighbor_cut_off_dis)
        num_connection_per_drone = num_drone_connection / self.num_drones
        num_neighbours_per_drone = num_drone_neighbours / self.num_drones

        # todo: use adjacency matrix to compute more stats about avoiding obstacles
        drone_obstacle_relation = self.agents_weighted_adj_matrix[self.drones_ids, self.obstacles_ids]
        num_obstacles_below_3_4 = np.count_nonzero(
            drone_obstacle_relation > 0 &
            drone_obstacle_relation < (self.drone_sensing_range_for_obstacles * 0.75)
        )
        num_obstacles_below_1_2 = np.count_nonzero(
            drone_obstacle_relation > 0 &
            drone_obstacle_relation < (self.drone_sensing_range_for_obstacles * 0.5)
        )
        num_obstacles_below_1_4 = np.count_nonzero(
            drone_obstacle_relation > 0 &
            drone_obstacle_relation < (self.drone_sensing_range_for_obstacles * 0.25)
        )
        # compute how many collision and impacts so far
        drones_events = self.global_state[_R_EVENT][:, self.drones_ids, :]
        total_collision = np.count_nonzero(drones_events == COLLISION_EVENT)
        total_impacts = np.count_nonzero(drones_events == IMPACT_EVENT)
        collision_per_step = total_collision / self.timestep
        impact_per_step = total_impacts / self.timestep
        collision_per_step_per_drone = collision_per_step / self.num_drones
        collision_per_step_per_drone_per_obstacle = collision_per_step_per_drone / self.num_obstacles
        impact_per_step_per_drone = impact_per_step / self.num_drones
        impact_per_step_per_drone_per_buildings = impact_per_step_per_drone / self.num_buildings

        info = {
            "goals captured so far": goals_captured_so_far,
            "number of steps for each capture": steps_per_goals_captured,
            "number of drones' connection": num_drone_connection,
            "number of drones' neighbours": num_drone_neighbours,
            "number of connection per drone": num_connection_per_drone,
            "number of neighbours per drone": num_neighbours_per_drone,
            "number of obstacles below 3/4 sensing range": num_obstacles_below_3_4,
            "number of obstacles below 1/2 sensing range": num_obstacles_below_1_2,
            "number of obstacles below 1/4 sensing range": num_obstacles_below_1_4,
            "number of total collision so far": total_collision,
            "number of collision/step": collision_per_step,
            "number of collision/(step, drone)": collision_per_step_per_drone,
            "number of collision/(step, drone, obstacles)": collision_per_step_per_drone_per_obstacle,
            "number of total impact so far": total_impacts,
            "number of impact/step": impact_per_step,
            "number of impact/(step, drone)": impact_per_step_per_drone,
            "number of impact/(step, drone, buildings)": impact_per_step_per_drone_per_buildings
        }
        return info

    def compute_reward(self):
        """
        Compute and return the rewards for each agent.
        """
        # initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        # we will calculate the rewards based on the start-end state

        # calculate flocking/separation reward based on neighborhood information
        for drone_id in self.drones_ids:
            _sep_reward_sum = 0
            _sep_num = 0
            _align_reward_sum = 0
            _align_num = 0
            coh_reward = 0
            _goal_rew = 0
            _neighbor_rel_dis_norm = self.agents_weighted_adj_matrix[drone_id]
            # get the neighbor agents (including goals) ids
            _all_neighbors = np.where(_neighbor_rel_dis_norm > 0)[0]

            # first compute impact and collision
            own_event = self.global_state[_R_EVENT][self.timestep][drone_id]
            num_impact = np.count_nonzero(own_event == IMPACT_EVENT)
            num_collision = np.count_nonzero(own_event == COLLISION_EVENT)
            impact_penalties = self.impact_penalty_coef * num_impact
            collision_penalties = self.collision_penalty_coef * num_collision
            rew[drone_id] -= (impact_penalties + collision_penalties)

            # iterate through the neighbors.
            # check goals as well
            if len(_all_neighbors) == 0:
                rew[drone_id] = 0
            else:
                own_speed = self.global_state[_SP][self.timestep][drone_id]
                own_v_speed = self.global_state[_V_SP][self.timestep][drone_id]
                own_yaw = self.global_state[_DIR][self.timestep][drone_id]
                own_loc = self.get_global_loc(drone_id)

                max_drone_goal_dis = self.linalg_norm(
                    np.array([self.env_size, self.env_size, self.env_max_height])
                )

                neighbor_loc = np.zeros(self.num_agents)
                num_neighbors = 0
                for agent_id in _all_neighbors:
                    rel_dis_norm = self.agents_weighted_adj_matrix[drone_id, agent_id]
                    other_loc = self.get_global_loc(agent_id)

                    if agent_id in self.drones_ids:
                        # this is a drone neighbor!
                        num_neighbors += 1
                        # update neighbourhood loc for cohesion later
                        neighbor_loc[agent_id] = other_loc
                        other_speed = self.global_state[_SP][self.timestep][agent_id]
                        other_v_speed = self.global_state[_V_SP][self.timestep][agent_id]
                        other_yaw = self.global_state[_DIR][self.timestep][agent_id]

                        # calculate the separation penalties
                        if rel_dis_norm < self.min_separation:
                            coef = self.separation_penalty_coef
                            _sep_reward_sum -= np.exp(- coef * rel_dis_norm)
                            _sep_num += 1

                        # calculate alignment within the neighborhood
                        if rel_dis_norm < self.neighbor_cut_off_dis:
                            coef = self.alignment_reward_coef
                            other_pitch = np.arctan2(other_v_speed, other_speed)
                            own_pitch = np.arctan2(own_v_speed, own_speed)
                            pitch_coh = np.exp(coef * (2 - np.cos(own_pitch - other_pitch)))
                            yaw_coh = np.exp(coef * (2 - np.cos(own_yaw - other_yaw)))
                            _align_reward_sum += (pitch_coh + yaw_coh)
                            _align_num += 1

                    if agent_id in self.goals_ids:
                        # compute goals relevant rewards
                        if self.capture_reward_mode is "Dense":
                            _goal_rew += - self.capture_reward_coef * \
                                         np.min((max_drone_goal_dis, rel_dis_norm)) / max_drone_goal_dis
                        if rel_dis_norm < self.capture_threshold:
                            self.still_in_the_game[agent_id] = 0
                            if self.capture_reward_mode is "sparse":
                                _goal_rew += self.capture_reward_sparse

                    rew[drone_id] += _goal_rew

                # cohesion
                if num_neighbors > 0:
                    neighbor_centroid = neighbor_loc / num_neighbors

                    drift_from_centroid = self.linalg_norm(
                        own_loc - neighbor_centroid
                    )
                    if drift_from_centroid >= self.min_centroid_drift:
                        coef = self.coherence_reward_coef
                        coh_reward += np.exp(- coef * drift_from_centroid)

                sep_penalty = _sep_reward_sum / _sep_num
                align_reward = _align_reward_sum / _align_num
                rew[drone_id] += (sep_penalty + align_reward + coh_reward)

        return rew

    # todo: set up the data dictionary to push to the device....
    # post this as issues on warp_drive repo, waiting for response..
    # if not get them in time, we may need to do some init in step function.
    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device.
        Note that these information would not be changed during the training.
        """
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _LOC_Z, _SP, _V_SP, _DIR, _ACC, _V_ACC, _SIG,
                        _R_LOC_X, _R_LOC_Y, _R_LOC_Z, _R_SP_X_, _R_SP_Y_, _R_SP_Z_, _R_EVENT]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )
        # todo: what should we set for logging_data_across_episode?
        # def: a data buffer of episode length is reserved for logging data
        # we may set the buildings_info and buildings_aabb as save&apply@reset
        # but we would modify it in-place as the first step of Numba step function(?)
        data_dict.add_data_list(
            [
                ("num_drones", self.num_drones, True),
                ("num_goals", self.num_goals, True),
                ("num_obstacles", self.num_obstacles, True),
                ("episode_length", self.episode_length, True),
                ("env_size", self.env_size, True),
                ("env_difficulty", self.env_difficulty, True),
                ("env_use_periodic_borders", self.env_use_periodic_borders, True),
                ("spacing_for_drones", self.spacing_for_drones, True),
                ("env_max_height", self.env_max_height, True),
                ("env_max_building_height", self.env_max_building_height, True),
                ("drones_ids", self.drones_ids, True),
                ("goals_ids", self.goals_ids, True),
                ("obstacles_ids", self.obstacles_ids, True),
                ("agent_type", self.agent_type, True),
                ("drones", self.drones, True),
                ("goals", self.goals, True),
                ("obstacles", self.obstacles, True),
                ("buildings_info", self.buildings_info, True),
                ("buildings_aabb", self.buildings_aabb, True),
                ("max_drone_speed", self.max_drone_speed, True),
                ("max_goal_speed", self.max_goal_speed, True),
                ("max_obstacle_speed", self.max_obstacle_speed, True),
                ("max_drone_vert_speed", self.max_drone_vert_speed, True),
                ("max_goal_vert_speed", self.max_goal_vert_speed, True),
                ("max_obstacle_vert_speed", self.max_obstacle_vert_speed, True),
                ("max_speed", self.max_speed, True),
                ("max_vert_speed", self.max_vert_speed, True),
                ("num_acceleration_levels", self.num_acceleration_levels, True),
                ("num_turn_levels", self.num_turn_levels, True),
                ("max_acceleration", self.max_acceleration, True),
                ("min_acceleration", self.min_acceleration, True),
                ("max_vert_acceleration", self.max_vert_acceleration, True),
                ("min_vert_acceleration", self.min_vert_acceleration, True),
                ("max_turn", self.max_turn, True),
                ("min_turn", self.min_turn, True),
                ("acceleration_actions", self.acceleration_actions, True),
                ("_acc_interval", self._acc_interval, True),
                ("vert_acceleration_actions", self.vert_acceleration_actions, True),
                ("turn_actions", self.turn_actions, True),
                ("_turn_interval", self._turn_interval, True),
                ("_action_space_array", self._action_space_array, True),
                ("use_communication", self.use_communication, True),
                ("drone_sensing_range", self.drone_sensing_range, True),
                ("drone_sensing_range_for_obstacles", self.drone_sensing_range_for_obstacles, True),
                ("drone_sensing_range_for_buildings", self.drone_sensing_range_for_buildings, True),
                ("neighbor_cut_off_dis", self.neighbor_cut_off_dis, True),
                ("min_separation", self.min_separation, True),
                ("min_centroid_drift", self.min_centroid_drift, True),
                ("separation_penalty_coef", self.separation_penalty_coef, True),
                ("alignment_reward_coef", self.alignment_reward_coef, True),
                ("coherence_reward_ceof", self.coherence_reward_coef, True),
                ("impact_penalty_coef", self.impact_penalty_coef, True),
                ("capture_threshold", self.capture_threshold, True),
                ("capture_reward_mode", self.capture_reward_mode, True),
                ("capture_reward_sparse", self.capture_reward_sparse, True),
                ("capture_reward_coef", self.capture_reward_coef, True),
                ("agent_radius", self.agents_radius, True),
                ("step_penalty_reward", self.step_penalty_reward, True),
                ("stab_penalty_coef", self.stab_penalty_coef, True),
                ("record_fps", self.record_fps, True)
            ]
        )

        pass

    def get_tensor_dictionary(self):
        """
        This data would be accessible to PyTorch during the training.

        """
        tensor_dict = DataFeed()
        return tensor_dict

    def reset(self):
        """
        Env reset().
        """
        # reset time to the beginning
        # todo: enhance the reproducibility of simulation by passing random seed
        # todo: for all the spawning process(?)
        self.timestep = 0
        self.drones_ids = self.np_random.choice(
            np.arange(self.num_agents), self.num_drones, replace=False
        )
        # initialize the drone local neighborhood
        self.agents_weighted_adj_matrix = {
            drone_id: [] for drone_id in self.drones_ids
        }
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

        self.max_speed = [
            self.max_drone_speed if self.agent_type[i] == DRONES else
            self.max_obstacle_speed if self.agent_type[i] == OBSTACLES else
            self.max_goal_speed for i in range(self.num_agents)
        ]
        self.max_vert_speed = [
            self.max_drone_vert_speed if self.agent_type[i] == DRONES else
            self.max_obstacle_vert_speed if self.agent_type[i] == OBSTACLES else
            self.max_goal_vert_speed for i in range(self.num_agents)
        ]
        drone_centroid = np.array(
            [self.env_size / 2 + 0.5, self.env_size / 2 + 0.5, self.env_max_height / 2 + 0.5],
            dtype=self.float_dtype
        )
        d_dict = spawn_drones_central(
            num_drones=self.num_drones,
            num_agents=self.num_agents,
            agent_type=self.agent_type,
            step_size=self.spacing_for_drones,
            drone_centroid=drone_centroid
        )
        g_dict = spawn_goals_padded(
            num_goals=self.num_goals,
            goals_ids=self.goals_ids,
            env_size=self.env_size,
            env_max_height=self.env_max_height
        )
        o_dict = spawn_obstacles_padded(
            env_size=self.env_size,
            max_height=self.env_max_height - 1,
            min_height=5,
            num_obstacles=self.num_obstacles,
            obstacles_ids=self.obstacles_ids
        )
        # todo: for Numba version, we only initialize it here
        if self.env_backend == "Numba":
            city_map = np.zeros((self.env_size + 2, self.env_size + 2))
        else:
            city_map = generate_city_padded(
                env_size=self.env_size,
                max_buildings_height=self.env_max_building_height,
                difficulty_level=self.env_difficulty,
                d_centroid=drone_centroid,
                d_dict=d_dict
            )

        # generate the city buildings' aabb info
        # todo: for Numba environment, we would only not initialize it
        #   but calculate it later
        if self.env_backend == "Numba":
            pass
        else:
            self.buildings_info, self.buildings_aabb, self.num_buildings = \
                compute_city_info(self.env_max_height, city_map)

        # init agents adjacency matrix with all -1
        self.agents_weighted_adj_matrix = np.full((self.num_agents, self.num_agents), -1)
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
        self.set_global_state(key=_R_LOC_X, value=None, t=self.timestep)
        self.set_global_state(key=_R_LOC_Y, value=None, t=self.timestep)
        self.set_global_state(key=_R_LOC_Z, value=None, t=self.timestep)
        self.set_global_state(key=_R_SP_X_, value=None, t=self.timestep)
        self.set_global_state(key=_R_SP_Y_, value=None, t=self.timestep)
        self.set_global_state(key=_R_SP_Z_, value=None, t=self.timestep)
        self.set_global_state(key=_R_EVENT, value=None, t=self.timestep)
        # array to keep track of the agents that are still in game
        self.still_in_the_game = np.ones(self.num_agents, dtype=self.int_dtype)

        # initialize global state for "still_in_the_game" to all ones
        self.global_state[_SIG] = np.ones(
            (self.episode_length + 1, self.num_agents), dtype=self.int_dtype
        )
        # Reinitialize variables that may have changed during previous episode

        # Calculate the speed x and speed y for the observations
        speed_x = self.starting_speeds * np.cos(self.starting_directions)
        speed_y = self.starting_speeds * np.sin(self.starting_directions)
        speed_z = self.starting_vert_speeds
        return self.generate_observation()

    def step_goals(self):
        """
        Step the goals such that they run away from the nearest observed drones.
        The generated actions does not guarantee to be collision free.
        Returns: the delta accelerations, delta_vert_acc, and directions.

        """
        capture_info = self.global_state[_SIG][self.timestep - 1]
        for goal_id in self.goals_ids:
            if capture_info[goal_id]:
                # re-spawning
                x_max = np.random.choice([0, 1], 1)[0] * self.env_size
                y_max = np.random.choice([0, 1], 1)[0] * self.env_size
                x = self.float_dtype(np.random.uniform(0, x_max))
                y = self.float_dtype(np.random.uniform(0, y_max))
                z = self.float_dtype(np.random.uniform(0, self.env_max_height-5))
                self.global_state[_LOC_X][self.timestep - 1][goal_id] = x
                self.global_state[_LOC_Y][self.timestep - 1][goal_id] = y
                self.global_state[_LOC_Z][self.timestep - 1][goal_id] = z
            # reset the goal to be in game
            # this would be pushed to global state in update_state
            self.still_in_the_game = np.ones(self.num_agents)

        # if a goal is out, spawn it at one of the city corner (need to check if there are any nearby drones)
        # if prescribed spawning positions do not satisfy the requirements, search along the borders to find one.
        # the border padding is always empty, waiting for spawning.

        # extract the drones prev location
        # note that the timestep has been updated to current one when called in self.step()
        prev_loc_x = self.global_state[_LOC_X][self.timestep - 1]
        prev_loc_y = self.global_state[_LOC_Y][self.timestep - 1]
        prev_loc_z = self.global_state[_LOC_Z][self.timestep - 1]
        prev_speed = self.global_state[_SP][self.timestep - 1]
        prev_v_speed = self.global_state[_V_SP][self.timestep - 1]
        prev_dir = self.global_state[_DIR][self.timestep - 1]
        prev_loc_temp = np.vstack((prev_loc_x, prev_loc_y, prev_loc_z))
        prev_loc = np.swapaxes(prev_loc_temp, 0, 1)
        g_action_list = []
        for goal_id in self.goals_ids:
            g_prev_loc = prev_loc[goal_id]
            g_prev_speed_ = prev_speed[goal_id] / self.env_size
            g_prev_v_speed_ = prev_v_speed[goal_id] / self.env_max_height
            g_prev_dir = prev_dir[goal_id]
            g_speed_feat = np.array([g_prev_speed_, g_prev_v_speed_, g_prev_dir])
            # compute the goal to drones distances
            drone_threat_list = []
            total_threat_score = 0
            for drone_id in self.drones_ids:
                d_prev_loc = prev_loc[drone_id]
                rel_dis = np.linalg.norm(g_prev_loc - d_prev_loc)
                if rel_dis < self.goal_sensing_range or self.use_full_observation:
                    # calculate its approaching speed
                    d_speed_ = prev_speed[drone_id] / self.env_size
                    d_v_speed_ = prev_v_speed[drone_id] / self.env_max_height
                    d_dir = prev_dir[drone_id]
                    d_speed_feat = np.array([d_speed_, d_v_speed_, d_dir])
                    # calculate the similarity
                    speed_feat_sim = np.dot(g_speed_feat, d_speed_feat)
                    # use normalized loc to calculate loc similarity
                    g_loc_feat = g_prev_loc / self.env_size
                    d_loc_feat = d_prev_loc / self.env_size
                    loc_feat_sim = np.dot(d_loc_feat, g_loc_feat)
                    # higher speed similarity -> less threat
                    s_threat_score = - speed_feat_sim
                    # higher loc similarity -> more threat
                    l_threat_score = loc_feat_sim
                    _threat_score = s_threat_score + l_threat_score
                    drone_threat_list.append([_threat_score, drone_id])
                    total_threat_score += _threat_score

            # if the threat list is empty
            if len(drone_threat_list) == 0:
                # just sample!
                g_action = self._action_space.sample()
            else:
                # we want to have a speed very similar to the chasing drone
                expected = 0
                g_prev_speed = np.array([prev_speed[goal_id], prev_v_speed[goal_id], prev_dir[goal_id]])
                for threat_score, drone_id in drone_threat_list:
                    weight = threat_score / total_threat_score
                    d_speed = prev_speed[drone_id]
                    d_v_speed = prev_v_speed[drone_id]
                    d_prev_dir = prev_dir[drone_id]
                    contribution = [d_speed, d_v_speed, d_prev_dir] * weight
                    expected += contribution
                diff = expected - g_prev_speed
                expected_acc, expected_v_acc, expected_dir = diff
                clipped_acc = np.clip(expected_acc, self.min_acceleration, self.max_acceleration)
                clipped_v_acc = np.clip(expected_v_acc, self.min_acceleration, self.max_acceleration)
                # note that the dir may not have the same range as max_/min turn (-pi/2, pi/2)
                # but that is a limitation anyway, so it is fine.
                clipped_turn = np.clip(expected_dir, self.min_turn, self.max_turn)
                # map the action back to action id
                acc_action_id = np.where(
                    np.logical_and(
                        self.acceleration_actions >= clipped_acc,
                        self.acceleration_actions <= clipped_acc + self._acc_interval
                    )
                )
                v_acc_action_id = np.where(
                    np.logical_and(
                        self.vert_acceleration_actions >= clipped_v_acc,
                        self.vert_acceleration_actions <= clipped_v_acc + self._acc_interval
                    )
                )
                turn_action_id = np.where(
                    np.logical_and(
                        self.turn_actions >= clipped_turn,
                        self.turn_actions <= clipped_turn + self._turn_interval
                    )
                )
                g_action = np.array([acc_action_id, v_acc_action_id, turn_action_id])
            g_action_list.append(g_action)

        return g_action_list

    def step_obstacles(self):
        """
        Randomly step the obstacles.Just sample from the action space.
        Returns: the delta accelerations, delta_vert_acc, and directions.

        """
        obstacles_action = []
        for obstacles in range(self.num_obstacles):
            _action = self._action_space.sample()
            obstacles_action.append(_action)
        return obstacles_action

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
            assert isinstance(actions, np.ndarray) or isinstance(actions, dict)
            if isinstance(actions, dict):
                # first sort the dict
                actions = {_id: act for _id, act in sorted(actions.items())}
                actions = np.array(list(actions.values()), dtype=int_dtype)
            # get the actions list and cast them to numpy arrays
            all_actions = np.empty((self.num_agents, 3))  # shape: (num_agents, 3)
            obstacles_actions = np.array(self.step_obstacles())  # shape: (num_obs, 3)
            goals_actions = np.array(self.step_goals())  # shape: (num_goals, 3)
            drones_actions = np.array(actions)  # shape: (num_drones, 3)
            assert obstacles_actions.shape == (self.num_obstacles, 3)
            assert goals_actions.shape == (self.num_goals, 3)
            assert drones_actions.shape == (self.num_drones, 3)
            all_actions[self.drones_ids] = drones_actions
            all_actions[self.obstacles_ids] = obstacles_actions
            all_actions[self.goals_ids] = goals_actions

            acc_action_ids = self.int_dtype(all_actions[:, 0])
            vert_acc_action_ids = self.int_dtype(all_actions[:, 1])
            turn_action_ids = self.int_dtype(all_actions[:, 2])

            assert np.all(
                (acc_action_ids >= 0) & (acc_action_ids <= self.num_acceleration_levels)
            )
            assert np.all(
                (vert_acc_action_ids >= 0) & (vert_acc_action_ids <= self.num_acceleration_levels)
            )
            assert np.all(
                (turn_action_ids >= 0) & (turn_action_ids <= self.num_turn_levels)
            )
            delta_accelerations = self.acceleration_actions[acc_action_ids]
            delta_vert_acc = self.vert_acceleration_actions[vert_acc_action_ids]
            delta_turns = self.turn_actions[turn_action_ids]

            # Update state and generate observation
            observation, reward, info = self.update_state(delta_accelerations, delta_vert_acc, delta_turns)

            if self.timestep >= self.episode_length:
                truncated = True
            else:
                truncated = False

            return observation, reward, info, truncated
        # return result
        pass
