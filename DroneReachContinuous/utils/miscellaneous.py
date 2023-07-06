import math

import numpy as np
import torch
from torch.distributions import Categorical

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

# todo: do we need to have diagonal adjacency for up and down?
adjacent_walk = np.array(
    [
        [1, 0, 0],  # front
        [-1, 0, 0],  # back
        [0, 1, 0],  # right
        [0, -1, 0],  # left
        [0, 0, 1],  # up
        [0, 0, -1],  # down
        [1, 1, 0],  # front-right
        [-1, -1, 0],  # back-left
        [1, -1, 0],  # front-left
        [-1, 1, 0]  # back-right
    ], dtype=int_dtype
)

spawn_cycle = even_spawn_rule.shape[0]


def stacking_maps(map_size, buildings_map, obstacles_map):
    """Given two numpy arrays of (size, size, size), stack them at each layer."""
    stacked_state = np.zeros((map_size, map_size, map_size))
    for i in range(map_size - 1):
        stacked_state[i, :, :] = buildings_map[i, :, :] + obstacles_map[i, :, :]
    return stacked_state


def create_actor_distribution(actor_output, action_size):
    """Create a distribution that actor can then use to randomly draw actions"""
    """Only used for discrete actions"""
    assert actor_output.size()[1] == action_size, "Actor output the wrong size"
    action_distribution = Categorical(actor_output)

    return action_distribution


def p_dict_to_block_list(data):
    """
    Given a dictionary whose values are numpy array representing points coordinates,
    calculate blocks number that holds these points. Assume that block index starts
    from 0 and of size 1x1x1.
    :param data: a dictionary whose values are numpy array for points coordinates.
    :return: a numpy array of blocks number of type np.int32.
    """
    assert isinstance(data, dict)
    coords_source = data.values()
    loc_list = np.unique(
        [
            np.int32(coords - coords % 1) for coords in coords_source
        ],
        axis=0
    )
    return loc_list


def count_points_in_block(data_dict):
    """
    Given a dictionary whose values are numpy array representing points coordinates,
    calculate corresponding blocks number that holds these points, and count how many
    points are within these blocks.
    :param data_dict: a dictionary whose values are numpy array for points coordinates.
    :return: a dictionary, key being blocks id in tuple, and values a list of points id that reside in this block.
    """
    assert isinstance(data_dict, dict)
    block_dict = {}
    for point_id in data_dict.keys():
        coords = data_dict[point_id]
        block_id = np.array(coords - coords % 1, dtype=int_dtype)
        block_id = tuple(block_id)
        if block_id not in block_dict.keys():  # new block, create the list to hold the point id
            block_dict[block_id] = [point_id]
        else:  # existing block, then append it to the existing list.
            block_dict[block_id].append(point_id)

    return block_dict


def generate_adjacent_blk(
        env_size: np.ndarray,
        block_id: np.ndarray):
    """
    Given a block id, generate the adjacent blocks id
    that are valid (within the boundaries...)
    :param env_size: a numpy array indicating the (x_bound, y_bound, z_bound) of the env
    :param block_id: a numpy array indicating blocks id
    :return: a list of adjacent blocks to the given block
    """

    x_min, y_min, z_min = -1, -1, -1
    x_max, y_max, z_max = env_size[0], env_size[1], env_size[2]

    possible_blocks = np.array([block_id + adjacent_walk[i] for i in range(len(adjacent_walk))], dtype=int_dtype)

    valid_blocks = [
        candidate_blk for candidate_blk in possible_blocks
        if x_min < candidate_blk[0] < x_max and
           y_min < candidate_blk[1] < y_max and
           z_min < candidate_blk[2] < z_max
    ]

    return valid_blocks


def get_building_dict(city_map: np.ndarray):
    """
    Given the city base map, convert it to a dictionary.
    :param city_map: a numpy array of shape (env_size, env_size) whose entries are buildings' height.
    :return: a dictionary whose keys are buildings location in tuple, and entry are buildings' height.
    """
    size = city_map.shape[0]
    buildings_dict = {}

    for i in range(size):
        for j in range(size):
            if city_map[i][j] > 0:
                buildings_dict[(i, j)] = city_map[i][j]

    return buildings_dict


def Bresenham3D_batch(
        x1_batch: np.ndarray,
        y1_batch: np.ndarray,
        z1_batch: np.ndarray,
        x2_batch: np.ndarray,
        y2_batch: np.ndarray,
        z2_batch: np.ndarray):
    """
    Batch version of Bresenham3D function.
    Return a list of listed blocks between starts and ends
    """
    assert x1_batch.shape == x2_batch.shape
    assert y1_batch.shape == y2_batch.shape
    assert z1_batch.shape == z2_batch.shape
    assert x1_batch.shape == y1_batch.shape
    assert x1_batch.shape == z1_batch.shape
    assert y1_batch.shape == z1_batch.shape

    num_points = x1_batch.shape[0]
    return_List = []
    for id_ in range(num_points):
        ListOfPoints = Bresenham3D(
            x1_batch[id_],
            y1_batch[id_],
            z1_batch[id_],
            x2_batch[id_],
            y2_batch[id_],
            z2_batch[id_]
        )
        return_List.append(ListOfPoints)

    return return_List


def Bresenham3D(x1, y1, z1, x2, y2, z2):
    """
    Given coordinates of 2 blocks, calculate the coordinates of all blocks on the
    line joining them. *Only accept integer inputs*.
    Use Bresenham line-drawing algorithm in 3D space.
    :param x1: x of starting block
    :param y1: y of starting block
    :param z1: z of starting block
    :param x2: x of ending block
    :param y2: y of ending block
    :param z2: z of ending block
    :return: a list of blocks coordinates
    """
    ListOfPoints = [[x1, y1, z1]]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if x2 > x1:
        xs = 1
    else:
        xs = -1
    if y2 > y1:
        ys = 1
    else:
        ys = -1
    if z2 > z1:
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append([x1, y1, z1])

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append([x1, y1, z1])

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append([x1, y1, z1])
    return ListOfPoints


def polyak_average(net, target_net, tau=0.01):
    """Given two neural networks, perform a soft update of second one
    :param net: the more up-to-date networks
    :param target_net: the network to be updated
    :param tau: control how much the update is. 1 for total copy. 0 for no update.
    """
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)
