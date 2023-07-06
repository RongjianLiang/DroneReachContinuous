from mayavi import mlab
from tvtk.tools import visual
import numpy as np
import os
import subprocess

agents_color = [
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0)
]

SCALE = 1

DRONES = 1
GOALS = 0
OBSTACLES = 0.5


def Arrow_From_A_to_B(x1, y1, z1, x2, y2, z2):
    ar1 = visual.Arrow(x=x1, y=y1, z=z1)
    ar1.length_cone = 1

    arrow_length = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos/arrow_length
    ar1.axis = [x2-x1, y2-y1, z2-z1]
    return ar1


def mayavi_render_all(
        size: int,
        data: np.ndarray,
        d_dict: dict,
        g_dict: dict,
        o_dict: dict):
    """
    Rendering city/buildings with 3D bar-charts and all the agents.
    This rendering would interrupt with the flow of process.
    *Assume each building is placed at the center of the block*.
    :param size: size of the environment, or number of blocks along
        each side of the environment.
    :param data:  city base map, a numpy array of shape (size, size) with
        each entry indicating the height of the buildings at that locations.
    :param d_dict: a dictionary with keys being drone agent ids,
        and values being their coordinates.
    :param g_dict: a dictionary with keys being goal agent ids,
        and values being their coordinates.
    :param o_dict: a dictionary with keys being obstacle agent ids,
        and values being their coordinates.
    """

    assert isinstance(size, int)
    assert isinstance(data, np.ndarray)
    assert data.shape == (size, size)
    assert isinstance(d_dict, dict)
    assert isinstance(g_dict, dict)
    assert isinstance(o_dict, dict)

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    x, y = np.mgrid[0.5: size - 0.5: size * 1j, 0.5: size - 0.5: size * 1j]
    mlab.barchart(x, y, data, colormap='inferno')
    mlab.vectorbar(title='Buildings height', nb_labels=10)

    d_coords = np.array([coord for coord in d_dict.values()])
    g_coords = np.array([coord for coord in g_dict.values()])
    o_coords = np.array([coord for coord in o_dict.values()])

    points = [d_coords, g_coords, o_coords]

    for agents_group in range(len(points)):
        x, y, z = np.transpose(points[agents_group])
        mlab.points3d(
            x, y, z, color=agents_color[agents_group], scale_factor=SCALE
        )

    axes = mlab.orientation_axes()
    axes.text_property.font_size = 8
    axes.text_property.justification = 'centered'
    mlab.view(
        azimuth=-60,
        elevation=45,
        distance=60,
        focalpoint=(10, 10, 10))
    mlab.show()


def mayavi_animate_series(
        size: int,
        total_time_step: int,
        num_agents: int,
        city: np.ndarray,
        visualize_border_type: list,
        agents_type: dict,
        agents_loc_x_series: np.ndarray,
        agents_loc_y_series: np.ndarray,
        agents_loc_z_series: np.ndarray,
        sphere_scale: float,
        cam_azimuth: int,
        cam_elevation: int,
        cam_distance: int,
        cam_foc,
        parallel_projection: bool,
        testing_cam: bool = False,
        visualize_borders: bool = False,
        pic_prefix: str = 'ani',
        _fps: int = 24,
        out_path: str = "./",
        pic_ext: str = '.png'):
    """
    A mayavi rendering function that accepts a fixed number of frames for each second
    of rendering. Expect input format like recorders in step_spheres.
    Args:
        visualize_border_type: a list of string ("up", "down", "left", "right", "top", "bottom")
        visualize_borders:
        testing_cam:
        parallel_projection:
        sphere_scale:
        size:
        total_time_step:
        num_agents:
        city:
        agents_type:
        agents_loc_x_series:
        agents_loc_y_series:
        agents_loc_z_series:
        cam_azimuth:
        cam_elevation:
        cam_distance:
        cam_foc:
        out_path:
        _fps:
        pic_prefix:
        pic_ext:

    Returns:

    """
    assert isinstance(size, int)
    assert isinstance(city, np.ndarray)
    assert city.shape == (size, size)
    assert isinstance(agents_type, dict)
    assert isinstance(agents_loc_x_series, np.ndarray)
    assert isinstance(agents_loc_y_series, np.ndarray)
    assert isinstance(agents_loc_z_series, np.ndarray)
    assert agents_loc_x_series.shape == (total_time_step * _fps, num_agents)
    assert agents_loc_y_series.shape == (total_time_step * _fps, num_agents)
    assert agents_loc_z_series.shape == (total_time_step * _fps, num_agents)

    out_path = "./"
    out_path = os.path.abspath(out_path)
    fps = _fps
    SCALE = sphere_scale
    prefix = pic_prefix
    ext = pic_ext

    mlab.options.offscreen = True

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    city_x, city_y = np.mgrid[0.5: size - 0.5: size * 1j, 0.5: size - 0.5: size * 1j]

    mlab.barchart(city_x, city_y, city, colormap='inferno')
    # mlab.vectorbar(title='Buildings height', nb_labels=10)

    # get the initial coords from shape (num_step * 24, 3)
    time_step = 0
    agent_x = agents_loc_x_series[time_step]
    agent_y = agents_loc_y_series[time_step]
    agent_z = agents_loc_z_series[time_step]

    drones = [id_ for id_ in range(num_agents) if agents_type[id_] == DRONES]
    goals = [id_ for id_ in range(num_agents) if agents_type[id_] == GOALS]
    obstacles = [id_ for id_ in range(num_agents) if agents_type[id_] == OBSTACLES]

    drones_plt = mlab.points3d(
        agent_x[drones], agent_y[drones], agent_z[drones], color=agents_color[0],
        scale_factor=SCALE)

    goals_plt = mlab.points3d(
        agent_x[goals], agent_y[goals], agent_z[goals], color=agents_color[1],
        scale_factor=SCALE)

    obstacles_plt = mlab.points3d(
        agent_x[obstacles], agent_y[obstacles], agent_z[obstacles], color=agents_color[2],
        scale_factor=SCALE)

    if visualize_borders:
        border_blocks = np.zeros((size + 2, size + 2, size + 2))
        s = size
        for border_type in visualize_border_type:
            if border_type == "top":
                border_blocks[0:s+2,        0:s+2,      s+1:s+2] = 1
            elif border_type == "bottom":
                border_blocks[0:s+2,        0:s+2,      0:1] = 1
            elif border_type == "north":
                border_blocks[0:1,          0:s+2,      0:s+2] = 1
            elif border_type == "south":
                border_blocks[s+1:s+2,      0:s+2,      0:s+2] = 1
            elif border_type == "west":
                border_blocks[0:s+2,        0:1,        0:s+2] = 1
            elif border_type == "east":
                border_blocks[0:s+2,        s+1:s+2,    0:s+2] = 1
            else:
                raise ValueError("Invalid border type")
        x, y, z = np.array(np.where(border_blocks == 1)) - 0.5
        mlab.points3d(x, y, z, mode="cube", scale_factor=1, color=(0, 1, 0), opacity=0.05)

    # # try to visualize the speed direction as well
    # v_start = np.array([
    #     agent_x, agent_y, agent_z
    # ])
    # v_end = np.array([
    #     agents_loc_x_series[time_step + 1],
    #     agents_loc_y_series[time_step + 1],
    #     agents_loc_z_series[time_step + 1]
    # ])
    # v_norm = np.linalg.norm(v_end - v_start, axis=1, keepdims=True)
    # v_arrows_obj = []
    # print(f"v_start: \n{v_start[0][0], v_start[1][0], v_start[2][0]}\n "
    #       f"v_end: \n{v_end[0][0], v_end[1][0], v_end[2][0]}\n, v_norm: \n{v_norm}\n")
    #
    # v = mlab.pipeline.vectors(mlab.pipeline.vector_scatter(
    #     v_start[0][0], v_start[1][0], v_start[2][0],
    #     v_end[0][0], v_end[2][0], v_end[1][0]
    # ))

    padding = len(str(agents_loc_x_series.shape[0]))
    # print(padding)
    d_s_plt = drones_plt.mlab_source
    g_s_plt = goals_plt.mlab_source
    o_s_plt = obstacles_plt.mlab_source

    num_frames = total_time_step * _fps

    def render_pics_and_save():
        f = mlab.gcf()
        for t in range(num_frames):
            f.scene.disable_render = True
            d_s_plt.set(
                x=agents_loc_x_series[t][drones],
                y=agents_loc_y_series[t][drones],
                z=agents_loc_z_series[t][drones]
            )

            g_s_plt.set(
                x=agents_loc_x_series[t][goals],
                y=agents_loc_y_series[t][goals],
                z=agents_loc_z_series[t][goals]
            )

            o_s_plt.set(
                x=agents_loc_x_series[t][obstacles],
                y=agents_loc_y_series[t][obstacles],
                z=agents_loc_z_series[t][obstacles]
            )
            f.scene.parallel_projection = parallel_projection
            f.scene.disable_render = False
            # f.scene.anti_aliasing_frames = 8
            # todo: create annotation for drones and goals using text3D
            # possible references:
            # https://stackoverflow.com/questions/12935231/annotating-many-points-with-text-in-mayavi-using-mlab
            # todo: create axes for the plot...
            axes = mlab.orientation_axes()
            # ax1 = mlab.axes(color=(0, 0, 0), nb_labels=size+1)
            axes.text_property.justification = 'centered'
            # todo: fix camera position
            mlab.view(
                azimuth=cam_azimuth,
                elevation=cam_elevation,
                distance=cam_distance,
                focalpoint=cam_foc)

            zeros = '0' * (padding - len(str(t)))
            filename = os.path.join(out_path, '{}_{}{}{}'.format(prefix, zeros, t, ext))
            mlab.savefig(filename=filename, size=(1280, 720))
            # print("saving pic: ", t)
            if testing_cam:
                break
            # break
        return

    render_pics_and_save()

    if not testing_cam:
        ffmpeg_fname = os.path.join(out_path, '{}_%0{}d{}'.format(prefix, padding, ext))
        cmd = 'ffmpeg -f image2 -r {} -i {}  ' \
              '-c:v libx264 -pix_fmt yuv420p -y {}.mp4'.format(fps, ffmpeg_fname, prefix)
        # print(out_path)
        # print(cmd)
        subprocess.check_output(['bash', '-c', cmd])

        # remove temp image files with extension
        [os.remove(f) for f in os.listdir(out_path) if f.endswith(ext)]


def mayavi_render_ffmpeg_video(
        size: int,
        max_time_step: int,
        num_agents: int,
        city: np.ndarray,
        agents_type: dict,
        agents_state: dict,
        cam_azimuth: int,
        cam_elevation: int,
        cam_distance: int,
        cam_foc,
        out_path: str = "./",
        _fps: int = 24,
        pic_prefix: str = 'ani',
        pic_ext: str = '.png'):
    """
    Save each time-step as a picture using mayavi and compile them into a video using ffmpeg.
    Args:
        cam_foc: camera focal point.
        cam_distance: camera viewing distance.
        cam_elevation: camera elevation.
        cam_azimuth: camera azimuth.
        pic_ext: picture format extension. Default to '.png'.
        pic_prefix: picture name prefix. Default to 'ani'
        _fps: frame per second for generating the video.
        out_path: output paths for the video, as well as temporary folder for images.
        size: size of the environment.
        max_time_step: total time step in the state dictionary.
        num_agents: total number of agents.
        city: city arrays.
        agents_type: a dictionary whose keys are agent ids, values are agent type indicator.
        agents_state: a dictionary with similar structure as global_state.
    """
    assert isinstance(size, int)
    assert isinstance(city, np.ndarray)
    assert city.shape == (size, size)
    assert isinstance(agents_type, dict)
    assert isinstance(agents_state, dict)

    out_path = "./"
    out_path = os.path.abspath(out_path)
    fps = _fps
    prefix = pic_prefix
    ext = pic_ext

    mlab.options.offscreen = True

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    city_x, city_y = np.mgrid[0.5: size - 0.5: size * 1j, 0.5: size - 0.5: size * 1j]

    mlab.barchart(city_x, city_y, city, colormap='inferno')
    # mlab.vectorbar(title='Buildings height', nb_labels=10)

    # get the initial coords
    time_step = 0
    agent_x = agents_state["loc_x"][time_step]
    agent_y = agents_state["loc_y"][time_step]
    agent_z = agents_state["loc_z"][time_step]

    drones = [id_ for id_ in range(num_agents) if agents_type[id_] == DRONES]
    goals = [id_ for id_ in range(num_agents) if agents_type[id_] == GOALS]
    obstacles = [id_ for id_ in range(num_agents) if agents_type[id_] == OBSTACLES]

    drones_plt = mlab.points3d(
        agent_x[drones], agent_y[drones], agent_z[drones], color=agents_color[0],
        scale_factor=SCALE)

    goals_plt = mlab.points3d(
        agent_x[goals], agent_y[goals], agent_z[goals], color=agents_color[1],
        scale_factor=SCALE)

    obstacles_plt = mlab.points3d(
        agent_x[obstacles], agent_y[obstacles], agent_z[obstacles], color=agents_color[2],
        scale_factor=SCALE)

    padding = len(str(len(agents_state["loc_x"])))
    # print(padding)
    d_s_plt = drones_plt.mlab_source
    g_s_plt = goals_plt.mlab_source
    o_s_plt = obstacles_plt.mlab_source

    # @mlab.animate(delay=100)
    def render_pics_and_save():
        f = mlab.gcf()
        for t in range(max_time_step):
            f.scene.disable_render = True
            d_s_plt.set(
                x=agents_state["loc_x"][t][drones],
                y=agents_state["loc_y"][t][drones],
                z=agents_state["loc_z"][t][drones]
            )

            g_s_plt.set(
                x=agents_state["loc_x"][t][goals],
                y=agents_state["loc_y"][t][goals],
                z=agents_state["loc_z"][t][goals]
            )

            o_s_plt.set(
                x=agents_state["loc_x"][t][obstacles],
                y=agents_state["loc_y"][t][obstacles],
                z=agents_state["loc_z"][t][obstacles]
            )

            f.scene.disable_render = False
            # f.scene.anti_aliasing_frames = 8
            # todo: create annotation for drones and goals using text3D
            # possible references:
            # https://stackoverflow.com/questions/12935231/annotating-many-points-with-text-in-mayavi-using-mlab
            # todo: create axes for the plot...
            axes = mlab.orientation_axes()
            axes.text_property.justification = 'centered'
            # todo: fix camera position
            mlab.view(
                azimuth=cam_azimuth,
                elevation=cam_elevation,
                distance=cam_distance,
                focalpoint=cam_foc)

            zeros = '0' * (padding - len(str(t)))
            filename = os.path.join(out_path, '{}_{}{}{}'.format(prefix, zeros, t, ext))
            mlab.savefig(filename=filename, size=(1280, 720))
            # print("saving pic: ", t)
        return

    render_pics_and_save()

    ffmpeg_fname = os.path.join(out_path, '{}_%0{}d{}'.format(prefix, padding, ext))
    cmd = 'ffmpeg -f image2 -r {} -i {}  ' \
          '-c:v libx264 -pix_fmt yuv420p -y {}.mp4'.format(fps, ffmpeg_fname, prefix)
    # print(out_path)
    # print(cmd)
    subprocess.check_output(['bash', '-c', cmd])

    # remove temp image files with extension
    [os.remove(f) for f in os.listdir(out_path) if f.endswith(ext)]


def render_city(
        size: int,
        data: np.ndarray
):
    assert isinstance(size, int)
    assert isinstance(data, np.ndarray)
    assert data.shape == (size, size)

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    x, y = np.mgrid[0.5: size - 0.5: size * 1j, 0.5: size - 0.5: size * 1j]
    mlab.barchart(x, y, data, colormap='inferno')
    mlab.vectorbar(title='Buildings height', nb_labels=10)

    mlab.show()


def render_with_adj_test(
        size: int,
        data: np.ndarray,
        adj: dict,
        d_dict: dict,
        g_dict: dict,
        o_dict: dict,
):
    assert isinstance(size, int)
    assert isinstance(data, np.ndarray)
    assert data.shape == (size, size)
    assert isinstance(adj, dict)
    assert isinstance(d_dict, dict)
    assert isinstance(g_dict, dict)
    assert isinstance(o_dict, dict)

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    x, y = np.mgrid[0.5: size - 0.5: size * 1j, 0.5: size - 0.5: size * 1j]
    mlab.barchart(x, y, data, colormap='inferno')
    mlab.vectorbar(title='Buildings height', nb_labels=10)

    d_coords = np.array([coord for coord in d_dict.values()])
    g_coords = np.array([coord for coord in g_dict.values()])
    o_coords = np.array([coord for coord in o_dict.values()])

    points = [d_coords, g_coords, o_coords]

    for agents_group in range(len(points)):
        x, y, z = np.transpose(points[agents_group])
        mlab.points3d(
            x, y, z, color=agents_color[agents_group], scale_factor=SCALE
        )

    # rendering adjacent blocks with high transparency
    # first color the central occ blocks
    for block_loc in adj.keys():
        block = np.zeros((size, size, size))
        block[block_loc[0], block_loc[1], block_loc[2]] = 1
        xx, yy, zz = np.array(np.where(block == 1)) + np.array(0.5, dtype=np.float32)  # stagger the coordinates
        mlab.points3d(xx, yy, zz, mode='cube', color=(0.5, 0, 1), opacity=0.5)

        # then color the adj block
        for adj_block in adj[block_loc]:
            adj_block_init = np.zeros((size, size, size))
            adj_block_init[adj_block[0], adj_block[1], adj_block[2]] = 1
            x_, y_, z_ = np.array(np.where(adj_block_init == 1)) + np.array(0.5, dtype=np.float32)
            mlab.points3d(x_, y_, z_, mode='cube', color=(0.5, 1, 0), opacity=0.5)

    mlab.show()
