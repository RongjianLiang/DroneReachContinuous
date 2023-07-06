import numpy as np

from utils.miscellaneous import Bresenham3D, p_dict_to_block_list

float_dtype = np.float32
int_dtype = np.int32

SEC_NUM = 2
MIN_MEAN_HEIGHT_PERCENTAGE = 0.3
MAX_MEAN_HEIGHT_PERCENTAGE = 0.7
NUM_INTERVAL = 10
HEIGHT_VARIANCE = 5.5

coords2blocks = np.array(
    [-0.5, -0.5, -0.5], dtype=float_dtype
)


def generate_city(
        env_size: int,
        max_buildings_height: int,
        difficulty_level: int,
        is_directly_obs: bool,
        d_centroid: np.ndarray,
        d_dict: dict,
        o_dict: dict,
        g_dict: dict, ):
    """
    Generate a city with buildings of various heights inside the environment, and
    satisfy various requirements including not interfering with agents' initial location,
    as well as if the goal(s) is/are in the line of sight of the drone formation centroid.
    :param env_size: size of the environment.
    :param max_buildings_height: maximum building height. It is suggested to have
        max_buildings_height < max_env_height.
    :param difficulty_level: an integer value indicating the cluster-ness of the buildings.
        Valid range is from 1~9.
        Although floating point number might make sense, it would not make
        too much a difference.
    :param is_directly_obs: if the agents have direct line of sight of the goal.
        Line of sight is determined from the centroid of drone formation.
    :param d_centroid: a numpy array indicating drone formation centroid coordinates.
    :param d_dict: a dictionary whose keys are drone agents ids,
        and values being their coordinates.
    :param o_dict: a dictionary whose keys are obstacles agents ids,
        and values being their coordinates.
    :param g_dict: a dictionary whose keys are goal agents ids,
        and values being their coordinates.
    :return: a numpy array of shape (env_size, env_size) with each entry indicating
        the height of the building at that coordinate.
    """
    assert isinstance(env_size, int)
    assert env_size % SEC_NUM == 0
    env_size = int_dtype(env_size)
    assert isinstance(max_buildings_height, int)
    max_buildings_height = int_dtype(max_buildings_height)
    assert isinstance(difficulty_level, int)
    assert isinstance(is_directly_obs, bool)
    assert isinstance(d_dict, dict)
    assert isinstance(o_dict, dict)
    assert isinstance(g_dict, dict)

    d_blk = p_dict_to_block_list(d_dict)
    g_blk = p_dict_to_block_list(g_dict)
    o_blk = p_dict_to_block_list(o_dict)

    h_m_list = np.linspace(MIN_MEAN_HEIGHT_PERCENTAGE, MAX_MEAN_HEIGHT_PERCENTAGE, NUM_INTERVAL)
    h_m = np.random.choice(h_m_list, SEC_NUM * SEC_NUM)
    h_s = HEIGHT_VARIANCE
    sec_size = int_dtype(env_size / SEC_NUM)
    num_per_sec = float_dtype(difficulty_level / 10)
    b_num = int_dtype(num_per_sec * sec_size * sec_size)
    h_min = np.zeros(b_num, dtype=int_dtype)
    h_max = np.full(b_num, max_buildings_height, dtype=int_dtype)

    d_centroid_blk = int_dtype(d_centroid + coords2blocks)
    g_centroid_blk = g_blk[0]  # basically just "unwrap" the list

    dx, dy, dz = d_centroid_blk
    gx, gy, gz = g_centroid_blk

    lineOfSight = Bresenham3D(dx, dy, dz, gx, gy, gz)

    occ_blk_list = np.concatenate(
        (d_blk, g_blk, o_blk)
    )

    buf = np.zeros((SEC_NUM * SEC_NUM, sec_size, sec_size), dtype=int_dtype)

    for sec_id in range(SEC_NUM * SEC_NUM):
        b_h = np.clip(
            np.random.normal(h_m[sec_id], h_s, b_num), h_min, h_max
        )
        b_h_pad = np.concatenate(
            (b_h, np.zeros(sec_size * sec_size - b_num))
        )
        np.random.shuffle(b_h_pad)
        b_h_arr = np.reshape(b_h_pad, (sec_size, sec_size))
        buf[sec_id] = b_h_arr

    sec_0_1 = np.hstack((buf[0], buf[1]))
    sec_2_3 = np.hstack((buf[2], buf[3]))
    c_b_m = np.vstack((sec_0_1, sec_2_3)).astype(int_dtype)

    envelope = np.full((env_size, env_size), max_buildings_height, dtype=int_dtype)
    for (x, y, z) in occ_blk_list:
        if envelope[x, y] > z:  # only update the entry if there is a smaller one.
            envelope[x, y] = z - 1

    if is_directly_obs:
        for (x, y, z) in lineOfSight:
            envelope[x, y] = z - 1

    else:  # make sure the sight is blocked. modify with minimal changes
        for (x, y, z) in lineOfSight:
            c_b_m[x, y] = np.where(c_b_m[x, y] > z, c_b_m[x, y], z + 1)

    c_b_m = np.where(c_b_m < envelope, c_b_m, envelope)

    return c_b_m
