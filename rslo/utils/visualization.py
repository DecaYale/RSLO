import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from io import StringIO, BytesIO
import PIL
import cv2
import rslo.utils.pose_utils_np as pun


def pltfig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    # fig.canvas.draw()

    # # Get the RGBA buffer from the figure
    # w, h = fig.canvas.get_width_height()
    # buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    # buf.shape = (w, h, 4)

    # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis=2)
    # buf = buf.astype(float)/255

    # 申请缓冲地址
    buffer_ = BytesIO()  # StringIO()  # using buffer,great way!
    # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
    fig.savefig(buffer_, format='png')
    buffer_.seek(0)
    # 用PIL或CV2从内存中读取
    dataPIL = PIL.Image.open(buffer_)
    # 转换为nparrary，PIL转换就非常快了,data即为所需
    data = np.asarray(dataPIL)
    data = data.astype(float)/255.
    # cv2.imwrite('test.png', data)
    # 释放缓存
    buffer_.close()
    plt.close(fig)

    return data


# def draw_odometry(odom_vectors, gt_vectors=None, view='bv', saving_dir=None):

def draw_trajectory(poses_pred, poses_gt=None, view='bv', saving_dir=None, figure=None, ax=None, color='b', error_step=1, odom_errors=None):
    """[summary]

    Arguments:
        poses_pred {[np.array]} -- [(N,7)]

    Keyword Arguments:
        poses_gt {[np.array]} -- [(N,7)] (default: {None})
        view {str} -- [description] (default: {'bv'})
        saving_dir {[type]} -- [description] (default: {None})
        figure {[type]} -- [description] (default: {None})
        ax {[type]} -- [description] (default: {None})
        color {str} -- [description] (default: {'b'})
    """

    assert(view in ['bv', 'front', 'side'])
    translation, rotation = poses_pred[:, :3], poses_pred[:, 3:]
    if poses_gt is not None:
        assert len(poses_pred) == len(poses_gt)
        translation_gt, rotation_gt = poses_gt[:, :3], poses_gt[:, 3:]

    if view == 'bv':
        dim0, dim1 = 0, 1
    elif view == 'front':
        dim0, dim1 = 0, 1
    elif view == 'side':
        dim0, dim1 = 0, 1

    if figure is None or ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(111)

    for i in range(1, len(translation)):
        if i == 1:
            ax.plot([translation[i-1][dim0]], [
                translation[i-1][dim1]], '*', markersize=10, color=color)

        ax.plot([translation[i-1][dim0], translation[i][dim0]], [
                translation[i-1][dim1], translation[i][dim1]], '-', markersize=0.5, color=color)

        if poses_gt is not None:
            ax.plot([translation_gt[i-1][dim0], translation_gt[i][dim0]], [
                translation_gt[i-1][dim1], translation_gt[i][dim1]], '-', markersize=0.5, color='r')
            if i % 50 == 0:
                # plot connection lines
                ax.plot([translation[i][dim0], translation_gt[i][dim0]], [
                    translation[i][dim1], translation_gt[i][dim1]], '-', markersize=0.03, color='gray')

    # and i%error_step==0 and i//error_step<len(errors):
    if 0:#odom_errors is not None:
        odom_errors = odom_errors[::error_step]
        l  = min(len(translation[::error_step] ), len(odom_errors))
        cm = plt.cm.get_cmap('hot')
        ax.scatter(translation[::error_step, dim0][:l]+10, translation[::error_step, dim1][:l]+10,
                    marker='o', c=odom_errors[:, 0], cmap=cm, vmin=np.min(odom_errors[:, 0]), vmax=np.max(odom_errors[:, 0]), linewidths=0.01)
        ax.scatter(translation[::error_step, dim0][:l]-10, translation[::error_step, dim1][:l]-10,
                    marker='x', c=odom_errors[:, 1], cmap=cm, vmin=np.min(odom_errors[:, 1]), vmax=np.max(odom_errors[:, 1]), linewidths=0.01)
        
    if saving_dir is not None:
        figure.savefig(saving_dir)

    return figure, ax


def draw_odometry(odom_vectors, view='bv', saving_dir=None, figure=None, ax=None, color='b'):
    """[draw  odometry]

    Args:
        odom_vectors ([numpy arrays of size (N,7)]): quaternion+translation
        gt (as the same as odom_vectors, optional): Defaults to None.
        view([str], optional): The view to draw
    """
    assert(view in ['bv', 'front', 'side'])

    translation, rotation = odom_vectors[:, :3], odom_vectors[:, 3:]

    if view == 'bv':
        # translation = translation[:, [0, 2]]
        dim0, dim1 = 0, 1
        # translation = translation[:, [0, 1]]
    elif view == 'front':
        dim0, dim1 = 0, 1

        # translation = translation[:, [0, 1]]
    elif view == 'side':
        dim0, dim1 = 0, 1
        # translation = translation[:, [1, 2]]

    if figure is None or ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(111)
    # lines = np.stack([starts, ends], axis=1)
    # lc = mc.LineCollection(lines, linewidths=0.3)
    # ax.add_collection(lc)
    t_prev = translation[0:1]
    r_prev = rotation[0:1]
    for i in range(1, len(translation)):
        r_cur = pun.qmult(r_prev, rotation[i:i+1])
        t_cur = t_prev + \
            pun.rotate_vec_by_q(
                translation[i:i+1], r_prev)
        # t_cur = translation[i]
        if i == 1:
            ax.plot([t_prev[0][dim0], t_cur[0][dim0]], [
                t_prev[0][dim1], t_cur[0][dim1]], '*', markersize=10, color=color)

        ax.plot([t_prev[0][dim0], t_cur[0][dim0]], [
                t_prev[0][dim1], t_cur[0][dim1]], '-', markersize=0.5, color=color)

        t_prev = t_cur
        r_prev = r_cur

    if saving_dir is not None:
        figure.savefig(saving_dir)

    return figure, ax
