from collision import *
from rendering import *


def multi_stepping(
        _fps: int,
        total_time_step: int,
        city_size: int,
        city_height: int,
        num_s: int,
        s_loc1: list,
        s_speed1: list,
        bs_info,
        bs_aabb,
        s_radius):

    total_loc_x = []
    total_loc_y = []
    total_loc_z = []
    total_e_recorder = []
    for step in range(total_time_step):
        new_loc, new_speed, recorders = step_spheres(
            city_size=city_size,
            city_height=city_height,
            num_spheres=num_s,
            spheres_location1=s_loc1,
            spheres_speeds=s_speed1,
            buildings_info=bs_info,
            buildings_aabb=bs_aabb,
            spheres_radius=s_radius,
            fps=_fps
        )
        # update loc for next step
        print(f"multi-step: {step}, new loc: {new_loc}")
        s_loc1 = new_loc
        s_speed1 = new_speed
        # below is for rendering
        # data preprocessing
        loc_recorder, s_recorder, e_recorder = recorders
        loc_recorder = array(loc_recorder)
        e_recorder = array(e_recorder)
        temp_e_recorder = np.swapaxes(e_recorder, 1, 0)  # shape: (3, 24)
        total_e_recorder.append(temp_e_recorder)
        # transpose sub-arrays, or frame arrays so that we have loc_x, loc_y, loc_z for all agents in each frame
        loc_temp = np.transpose(loc_recorder, (0, 2, 1))
        # to get the loc_x, loc_y, loc_z for each frame step
        total_loc_x.append(loc_temp[:, 0, :])  # shape: (24, 3)
        total_loc_y.append(loc_temp[:, 1, :])
        total_loc_z.append(loc_temp[:, 2, :])

    total_loc_x = array(total_loc_x).reshape(total_time_step * _fps, num_s)     # (num_s, _fps, 3) -> (num_s * _fps, 3)
    total_loc_y = array(total_loc_y).reshape(total_time_step * _fps, num_s)     # (num_s, _fps, 3) -> (num_s * _fps, 3)
    total_loc_z = array(total_loc_z).reshape(total_time_step * _fps, num_s)     # (num_s, _fps, 3) -> (num_s * _fps, 3)
    return total_loc_x, total_loc_y, total_loc_z, total_e_recorder


def animate_border_impact1():
    # set up the initial
    case_name = "border_impact_side"
    _fps = 24
    num_s = 1
    city_size = 5
    city_height = 5
    city = np.zeros((city_size, city_size))
    agents_type = {0: 0}
    s_loc1 = [
        array([city_size / 2, 0, 0])
    ]
    s_speed1 = [
        5 * array([0.5, 0.5, 0])
    ]
    bs_info, bs_aabb = compute_city_info(5, city)
    s_radius = 0.1

    total_time_step = 5

    camera = {
        "cam_azimuth": 0,
        "cam_distance": 10,
        "cam_elevation": 0,
        "cam_foc": (city_size / 2, city_size / 2, city_height / 2)
    }
    t_loc_x, t_loc_y, t_loc_z, t_e_recorder = multi_stepping(
        _fps, total_time_step, city_size, city_height, num_s, s_loc1, s_speed1, bs_info, bs_aabb, s_radius
    )
    vis_border = ["north", "south", "east", "west"]
    mayavi_animate_series(
        size=city_size,
        total_time_step=total_time_step,
        num_agents=num_s,
        city=city,
        visualize_border_type=vis_border,
        agents_type=agents_type,
        agents_loc_x_series=t_loc_x,
        agents_loc_y_series=t_loc_y,
        agents_loc_z_series=t_loc_z,
        sphere_scale=s_radius,
        cam_azimuth=camera["cam_azimuth"],
        cam_distance=camera["cam_distance"],
        cam_elevation=camera["cam_elevation"],
        cam_foc=camera["cam_foc"],
        parallel_projection=False,
        testing_cam=False,
        visualize_borders=True,
        pic_prefix=case_name
    )


def animate_border_impact2():
    case_name = "border_impact_top_bottom"
    _fps = 24
    num_s = 1
    city_size = 5
    city_height = 5
    city = np.zeros((city_size, city_size))
    agents_type = {0: 0}
    s_loc1 = [
        array([city_size / 2, 0, city_height / 2])
    ]
    s_speed1 = [
        5 * array([0, 0.5, 0.5])
    ]
    bs_info, bs_aabb = compute_city_info(5, city)
    s_radius = 0.1
    total_time_step = 5
    camera = {
        "cam_azimuth": 0,
        "cam_distance": 10,
        "cam_elevation": 90,
        "cam_foc": (city_size / 2, city_size / 2, city_height / 2)
    }
    t_loc_x, t_loc_y, t_loc_z, t_e_recorder = multi_stepping(
        _fps, total_time_step, city_size, city_height, num_s, s_loc1, s_speed1, bs_info, bs_aabb, s_radius
    )
    vis_border = ["top", "bottom", "east", "west"]
    mayavi_animate_series(
        size=city_size,
        total_time_step=total_time_step,
        num_agents=num_s,
        city=city,
        visualize_border_type=vis_border,
        agents_type=agents_type,
        agents_loc_x_series=t_loc_x,
        agents_loc_y_series=t_loc_y,
        agents_loc_z_series=t_loc_z,
        sphere_scale=s_radius,
        cam_azimuth=camera["cam_azimuth"],
        cam_distance=camera["cam_distance"],
        cam_elevation=camera["cam_elevation"],
        cam_foc=camera["cam_foc"],
        parallel_projection=False,
        testing_cam=False,
        visualize_borders=True,
        pic_prefix=case_name
    )


def animate_building_impact1():
    case_name = "building_impact1"
    _fps = 24
    num_s = 1
    city_size = 5
    city_height = 5
    city = np.zeros((city_size, city_size))
    city[2, 3] = 3
    city[1, 4] = 3
    city[4, 2] = 3
    agents_type = {0: 0}
    s_loc1 = [
        array([city_size / 2, 0, city_height / 2])
    ]
    s_speed1 = [
        5 * array([0.5, 0.5, 0])
    ]
    bs_info, bs_aabb = compute_city_info(5, city)
    s_radius = 0.1
    total_time_step = 5
    camera = {
        "cam_azimuth": 0,
        "cam_distance": 10,
        "cam_elevation": 0,
        "cam_foc": (city_size / 2, city_size / 2, city_height / 2)
    }
    t_loc_x, t_loc_y, t_loc_z, t_e_recorder = multi_stepping(
        _fps, total_time_step, city_size, city_height, num_s, s_loc1, s_speed1, bs_info, bs_aabb, s_radius
    )
    vis_border = ["north", "south", "east", "west"]
    mayavi_animate_series(
        size=city_size,
        total_time_step=total_time_step,
        num_agents=num_s,
        city=city,
        visualize_border_type=vis_border,
        agents_type=agents_type,
        agents_loc_x_series=t_loc_x,
        agents_loc_y_series=t_loc_y,
        agents_loc_z_series=t_loc_z,
        sphere_scale=s_radius,
        cam_azimuth=camera["cam_azimuth"],
        cam_distance=camera["cam_distance"],
        cam_elevation=camera["cam_elevation"],
        cam_foc=camera["cam_foc"],
        parallel_projection=False,
        testing_cam=False,
        visualize_borders=True,
        pic_prefix=case_name
    )


def animate_building_impact2():
    case_name = "building_impact2"
    _fps = 24
    num_s = 1
    city_size = 5
    city_height = 5
    city = np.zeros((city_size, city_size))
    city[2, 2] = 4
    agents_type = {0: 0}
    s_loc1 = [
        array([city_size / 2, 0, city_height / 2])
    ]
    s_speed1 = [
        5 * array([0, 0.5, -0.5])
    ]
    bs_info, bs_aabb = compute_city_info(5, city)
    s_radius = 0.1
    total_time_step = 5
    camera = {
        "cam_azimuth": 0,
        "cam_distance": 10,
        "cam_elevation": 90,
        "cam_foc": (city_size / 2, city_size / 2, city_height / 2)
    }
    t_loc_x, t_loc_y, t_loc_z, t_e_recorder = multi_stepping(
        _fps, total_time_step, city_size, city_height, num_s, s_loc1, s_speed1, bs_info, bs_aabb, s_radius
    )
    # print(f" loc_x: \n{t_loc_x}\n loc_y: \n{t_loc_y}\n loc_z: \n{t_loc_z}\n")
    # for e in range(len(t_e_recorder)):
    #     print(f"step: {e}\n {t_e_recorder[e]}\n")
    vis_border = ["north", "east", "west"]
    mayavi_animate_series(
        size=city_size,
        total_time_step=total_time_step,
        num_agents=num_s,
        city=city,
        visualize_border_type=vis_border,
        agents_type=agents_type,
        agents_loc_x_series=t_loc_x,
        agents_loc_y_series=t_loc_y,
        agents_loc_z_series=t_loc_z,
        sphere_scale=s_radius,
        cam_azimuth=camera["cam_azimuth"],
        cam_distance=camera["cam_distance"],
        cam_elevation=camera["cam_elevation"],
        cam_foc=camera["cam_foc"],
        parallel_projection=False,
        testing_cam=False,
        visualize_borders=True,
        pic_prefix=case_name
    )


def animate_collision1():
    case_name = "spheres_collision1_60fps"
    _fps = 60
    num_s = 2
    city_size = 5
    city_height = 5
    city = np.zeros((city_size, city_size))
    agents_type = {0: 0, 1: 1}
    s_loc1 = [
        array([city_size / 2, 0, city_height / 2]),
        array([city_size / 2, city_size, city_height / 2])
    ]
    s_speed1 = [
        5 * array([0, 0.5, 0]),
        5 * array([0, -.5, 0])
    ]
    bs_info, bs_aabb = compute_city_info(5, city)
    s_radius = 0.1
    total_time_step = 5
    camera = {
        "cam_azimuth": 0,
        "cam_distance": 6,
        "cam_elevation": 90,
        "cam_foc": (city_size / 2, city_size / 2, city_height / 2)
    }
    t_loc_x, t_loc_y, t_loc_z, t_e_recorder = multi_stepping(
        _fps, total_time_step, city_size, city_height, num_s, s_loc1, s_speed1, bs_info, bs_aabb, s_radius
    )
    # print(f" loc_x: \n{t_loc_x}\n loc_y: \n{t_loc_y}\n loc_z: \n{t_loc_z}\n")
    # for e in range(len(t_e_recorder)):
    #     print(f"step: {e}\n {t_e_recorder[e]}\n")
    vis_border = ["north", "east", "west"]
    mayavi_animate_series(
        size=city_size,
        total_time_step=total_time_step,
        num_agents=num_s,
        city=city,
        visualize_border_type=vis_border,
        agents_type=agents_type,
        agents_loc_x_series=t_loc_x,
        agents_loc_y_series=t_loc_y,
        agents_loc_z_series=t_loc_z,
        sphere_scale=s_radius,
        cam_azimuth=camera["cam_azimuth"],
        cam_distance=camera["cam_distance"],
        cam_elevation=camera["cam_elevation"],
        cam_foc=camera["cam_foc"],
        parallel_projection=False,
        testing_cam=False,
        visualize_borders=True,
        pic_prefix=case_name,
        _fps=_fps
    )


def animate_collision2():
    case_name = "spheres_collision2_60fps"
    _fps = 60
    num_s = 2
    city_size = 5
    city_height = 5
    city = np.zeros((city_size, city_size))
    agents_type = {0: 0, 1: 1}
    s_loc1 = [
        array([city_size / 2, 0.4, city_height / 2 + 0.1]),
        array([city_size / 2, city_size - 0.4, city_height / 2 - 0.1])
    ]
    s_speed1 = [
        5 * array([0, 0.5, 0]),
        5 * array([0, -.5, 0])
    ]
    bs_info, bs_aabb = compute_city_info(5, city)
    s_radius = 0.4
    total_time_step = 2
    camera = {
        "cam_azimuth": 0,
        "cam_distance": 6,
        "cam_elevation": 90,
        "cam_foc": (city_size / 2, city_size / 2, city_height / 2)
    }
    t_loc_x, t_loc_y, t_loc_z, t_e_recorder = multi_stepping(
        _fps, total_time_step, city_size, city_height, num_s, s_loc1, s_speed1, bs_info, bs_aabb, s_radius
    )
    vis_border = ["north", "east", "west"]
    mayavi_animate_series(
        size=city_size,
        total_time_step=total_time_step,
        num_agents=num_s,
        city=city,
        visualize_border_type=vis_border,
        agents_type=agents_type,
        agents_loc_x_series=t_loc_x,
        agents_loc_y_series=t_loc_y,
        agents_loc_z_series=t_loc_z,
        sphere_scale=s_radius,
        cam_azimuth=camera["cam_azimuth"],
        cam_distance=camera["cam_distance"],
        cam_elevation=camera["cam_elevation"],
        cam_foc=camera["cam_foc"],
        parallel_projection=False,
        testing_cam=False,
        visualize_borders=True,
        pic_prefix=case_name,
        _fps=_fps
    )


if __name__ == "__main__":
    # call whatever animate functions you want to execute here.
    pass
