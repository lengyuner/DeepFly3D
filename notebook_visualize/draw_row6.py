# %load_ext autoreload
# %autoreload 2



import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from deepfly.GUI.CameraNetwork import CameraNetwork
from deepfly.GUI.Config import config_fly as config
from deepfly.GUI.util.os_util import *
from deepfly.GUI.util.plot_util import normalize_pose_3d
from deepfly.GUI.util.signal_util import *

plt.style.use('dark_background')
skeleton = config["skeleton"]



image_folder = './data/test/'
output_folder = 'output'
# image_folder = '/mnt/NAS/CLC/190409_SS30303-tdTomGC6fopt/Fly1/CO2xzGG/behData_004/images/'
calib = read_calib("./data/test/")

# cid2cidread, cidread2cid = read_camera_order(image_folder)
# camNet = CameraNetwork(image_folder=image_folder, cam_id_list=range(7), calibration=calib, cid2cidread=cid2cidread, num_images=100)
# cam_list = camNet.cam_list
# camNetLeft = CameraNetwork(image_folder=image_folder, cam_id_list=[0,1,2], cam_list=cam_list[:3], calibration=calib, cid2cidread=cid2cidread, num_images=100)
# camNetRight = CameraNetwork(image_folder=image_folder, cam_id_list=[4,5,6], cam_list=cam_list[4:7],calibration=calib, cid2cidread=cid2cidread, num_images=100)


cid2cidread, cidread2cid = read_camera_order(image_folder)
# camNet = CameraNetwork(image_folder=image_folder,output_folder=output_folder,
#                        cam_id_list=range(7), calibration=calib, cid2cidread=cid2cidread, num_images=1)
camNet = CameraNetwork(image_folder=image_folder,output_folder=output_folder,
                       cam_id_list=np.arange(7), calibration=calib, cid2cidread=cid2cidread, num_images=1)


# image_folder = './data/test/'
# calib = read_calib("./data/test/")
# output_folder = 'output'
# cid2cidread, cidread2cid = read_camera_order(image_folder)
# camNet_gt = CameraNetwork(image_folder=image_folder, output_folder=output_folder,
#                           cam_id_list=np.arange(config["num_cameras"]), calibration=calib,
#                           cid2cidread=cid2cidread, num_images=1)



# camNet = camNet_gt

cam_list = camNet.cam_list
camNetLeft = CameraNetwork(image_folder=image_folder,output_folder=output_folder,
                           cam_id_list=[0,1,2], cam_list=cam_list[:3], calibration=calib,
                           cid2cidread=cid2cidread, num_images=1)

camNetRight = CameraNetwork(image_folder=image_folder, output_folder=output_folder,
                            cam_id_list=[4,5,6], cam_list=cam_list[4:7], calibration=calib,
                            cid2cidread=cid2cidread, num_images=1)


d = pickle.load(open(glob.glob(os.path.join(image_folder, 'pose_result*.pkl'))[0], 'rb'))

camNetList = [camNetLeft, camNetRight, camNet]
camNetList = [camNet]
for cn in camNetList:
    for cam in cn:
        cam.points2d = d["points2d"][cam.cam_id, :]
for cn in camNetList[:-1]:
    cn.triangulate()
    cn.bundle_adjust()
    camNet.triangulate()




# image_folder = './data/test/'
# calib = read_calib("./data/test/")

cid2cidread, cidread2cid = read_camera_order(image_folder)
camNet_gt = CameraNetwork(image_folder=image_folder, output_folder=output_folder,
                          cam_id_list=np.arange(config["num_cameras"]), calibration=calib,
                          cid2cidread=cid2cidread, num_images=1)
# loading final results
d = pickle.load(open(glob.glob(os.path.join(image_folder, 'pose_result*.pkl'))[0], 'rb'))


for cam in camNet_gt:
    cam.points2d = d["points2d"][cam.cam_id, :]

camNet_gt.triangulate()
camNet_gt.points3d_m = normalize_pose_3d(camNet_gt.points3d_m, rotate=False)



img_id = 0
camera_id_list = [0, 1, 2, 3, 4, 5, 6]
fig, ax_list = plt.subplots(1, len(camera_id_list), figsize=(60, 60))
draw_joints = range(skeleton.num_joints)


for cam_id, ax in zip(camera_id_list, ax_list):
    # cam_id, ax = zip(camera_id_list, ax_list)
    pt = camNet_gt.points3d_m[img_id, :, :]-1.5
    pts2d = camNet_gt[cam_id].project(pt)
    # print('1')

    zorder = skeleton.get_zorder(camNet[cam_id].cam_id)
    thickness = [5] * skeleton.num_limbs
    colors_tmp = skeleton.colors.copy()
    for l in range(skeleton.num_limbs):
        if not skeleton.camera_see_limb(camNet[cam_id].cam_id, l):
            thickness[l] = 5
            colors_tmp[l] = (125, 125, 125)
    ax.imshow(camNet[cam_id].plot_2d(img_id=img_id, pts=pts2d, colors=colors_tmp, thickness=thickness, draw_joints=None,
                                     zorder=zorder))
    ax.axis('off')

plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())




# d = pickle.load(open(glob.glob(os.path.join(image_folder, 'heatmap*.pkl'))[0], 'rb'))



##$$


from deepfly.pose3d.procrustes.procrustes import procrustes,\
    apply_transformation
#,__procrustes


m_left = np.arange(0,15)
points3d_gt_left = camNet_gt.points3d_m[:,m_left].copy()
# points3d_pred_left = camNet.points3d_m[:,m_left].copy()
pts_t_left, tform = procrustes(pts=points3d_gt_left, template=points3d_gt_left, reflection='best', verbose=False,
                               joint=[skeleton.Tracked.BODY_COXA, skeleton.Tracked.COXA_FEMUR], return_transf=True)


# def procrustes(pts, template=None, reflection='best', verbose=False,
#                joint=(skeleton.Tracked.BODY_COXA, skeleton.Tracked.COXA_FEMUR), return_transf=False):

# for cam in camNet:
# pts_t_left, tform = __procrustes(pts=points3d_gt_left, template=points3d_gt_left, scaling=False, joint=[skeleton.Tracked.BODY_COXA, skeleton.Tracked.COXA_FEMUR], reflection='best', return_transf=True)
for cam in camNet:
    if cam.cam_id < 3:
        cam.set_R(np.dot(cam.R, tform["rotation"]))
        cam.set_tvec(cam.tvec - np.dot(cam.R, tform["translation"]))
print(tform)

# % matplotlib
# widget
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deepfly.GUI.util.plot_util import plot_drosophila_3d


def plt_3d(img_id):
    fig = plt.figure()
    # fig.canvas.layout.width = '500px'
    ax3d = fig.add_subplot(111, projection='3d')
    draw_joints = [j for j in range(skeleton.num_joints) if
                   skeleton.is_tracked_point(j, skeleton.Tracked.BODY_COXA) or skeleton.is_tracked_point(j,
                                                                                                         skeleton.Tracked.COXA_FEMUR) or skeleton.is_tracked_point(
                       j, skeleton.Tracked.FEMUR_TIBIA) or skeleton.is_tracked_point(j,
                                                                                     skeleton.Tracked.TIBIA_TARSUS) or skeleton.is_tracked_point(
                       j, skeleton.Tracked.TARSUS_TIP)]

    lim = 2
    # plot_drosophila_3d(ax3d, pts3d_proc[img_id].copy(), 1, draw_joints=draw_joints,
    #                    lim=lim)  # , colors=[(0,0,255)]*100)
    plot_drosophila_3d(ax3d, camNet_gt.points3d_m[img_id].copy(), 1, draw_joints=draw_joints,
                       lim=lim)  # colors=[(0,255,0)]*100)
    # plot_drosophila_3d(ax3d, normalize_pose_3d(camNet.points3d_m.copy(), rotate=True)[img_id,:], 1, draw_joints=draw_joints)#colors=[(255,0,0)]*100)


plt_3d(0)



camNet_gt.points3d_m[0][np.newaxis, :].copy().shape





# %matplotlib inline
from ipywidgets import interactive
interactive_plot = interactive(plt_3d, img_id=(0,14))
output = interactive_plot.children[-1]
output.layout.height = '1000px'
interactive_plot






img_id = 0
camNet = camNet
camera_id_list = [0,1,2,3,4,5,6]
fig, ax_list = plt.subplots(1,len(camera_id_list),figsize=(60,60))
draw_joints = range(skeleton.num_joints)
for cam_id, ax in zip(camera_id_list, ax_list):
    # pt = pts3d_proc[img_id,:,:]
    pt = camNet_gt.points3d_m[img_id, :, :] - 1.5
    pts2d = camNet[cam_id].project(pt)

    zorder = skeleton.get_zorder(camNet[cam_id].cam_id)
    thickness = [5]*skeleton.num_limbs
    colors_tmp = skeleton.colors.copy()
    for l in range(skeleton.num_limbs):
        if not skeleton.camera_see_limb(camNet[cam_id].cam_id, l):
            thickness[l] = 5
            colors_tmp[l] = (125,125,125)
    ax.imshow(camNet[cam_id].plot_2d(img_id=img_id, pts=pts2d, colors=colors_tmp, thickness=thickness, draw_joints=None, zorder=zorder))
    ax.axis('off')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())





plt.style.use('default')
# % matplotlib
# inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deepfly.GUI.util.plot_util import plot_drosophila_3d
from deepfly.GUI.util.plot_util import normalize_pose_3d

fontsize = 10#50
num_rows, num_cols = 6, 7
fig = plt.figure(figsize=(num_cols * 9, num_rows * 5))

gs1 = gridspec.GridSpec(num_rows, num_cols)
gs1.update(wspace=0, hspace=0)  # set the spacing between axes.

plt_list = []
# camNet_gt.points3d_m
pts_t = camNet_gt.points3d_m.copy()
# pts_t = pts3d_filter.copy()
tmp = pts_t[:, :, 1].copy()
pts_t[:, :, 1] = pts_t[:, :, 2].copy()
pts_t[:, :, 2] = tmp
pts_t[:, :, 2] *= -1
pts_t[:, :, 1] *= -1

pts_t = normalize_pose_3d(pts_t, normalize_median=True)
title_list = ["Camera\ {}".format(i) for i in range(1, 8)]
ylabel_list = ["Raw\ image", "Prob. map", "2D\ pose", r"Projected 3D pose", "3D\ pose"]
for i in range(num_rows * num_cols):
    r = int(i / num_cols)
    c = i % num_cols
    if r == 4:
        ax1 = plt.subplot(gs1[r:, c], projection='3d')
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    elif r < 4:
        ax1 = plt.subplot(gs1[r, c])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_aspect('equal')
        # plt.axis('off')

    if r == 0:
        ax1.set_title(r"$\bf{" + title_list[c] + "}$", fontsize=fontsize, fontname="Times New Roman Bold")
    if c == 0 and r == 3:
        ax1.set_ylabel(r"$\bf{3D\ pose}$", fontsize=fontsize)
    if c == 0 and r != 3 and r != 4 and r != 5:
        ax1.set_ylabel(r"$\bf{" + ylabel_list[r] + "}$", fontsize=fontsize)
    if c == 0 and r == 4:
        pass

    if r == 4:
        plt_list.append(ax1)
    elif r < 4:
        plt_list.append(ax1.imshow(np.zeros((480, 960, 3), dtype=np.uint8)))
        if c == num_cols - 1 and r == 0:
            pass

import cv2

# def get_image(self, img_id, flip=False):
#     image_path = os.path.join(self.image_folder,"camera_{}_img_{:06}.jpg".format(self.cam_id_read, img_id))
#     print('Reading: ' + image_path)##
#     img = cv2.imread(image_path)
#
#     print(image_path, '$'*100 )##
#     if img is None:
#         print("Cannot find", self.cam_id, img_id)
#     return img

# self = camNet[c]
# os.path.join(self.image_folder,"camera_{}_img_{:06}.jpg".format(self.cam_id_read, img_id))


img_id_list = np.arange(0, 1)
for img_id in img_id_list:
    # print(img_id)
    if img_id % 50 == 0:
        print(img_id)
    for r in range(num_rows):
        for c in range(num_cols):
            img = camNet[c].get_image(img_id)
            zorder = skeleton.get_zorder(c)
            draw_joints = [j for j in range(skeleton.num_joints) if skeleton.camera_see_joint(c, j)]
            i = r * num_cols + c
            if r == 0:  # raw
                plt_list[i].set_data(img)
            # elif r == 1:  # heatmap
            #     plt_list[i].set_data(
            #         camNet[c].plot_heatmap(img_id, img=img, flip_heatmap=False, draw_joints=draw_joints))
            elif r == 2:  # 2d pose
                draw_limbs = None
                plt_list[i].set_data(
                    camNet[c].plot_2d(img_id, img=img, flip_points=False, draw_limbs=draw_limbs, zorder=zorder,
                                      thickness=[10] * skeleton.num_limbs))
            elif r == 3:  # projection of 3d pose
                draw_joints = range(skeleton.num_joints)
                # pt = pts3d_filter[img_id, :, :]
                # camNet_gt.points3d_m
                pt = camNet_gt.points3d_m[img_id, :, :]
                for j in range(skeleton.num_joints):
                    if skeleton.is_tracked_point(j, skeleton.Tracked.STRIPE) and skeleton.is_joint_visible_left(j):
                        pt[j] = (pt[j] + pt[j + (skeleton.num_joints // 2)]) / 2
                        pt[j + skeleton.num_joints // 2] = pt[j]
                pts2d = camNet[c].project(pt)

                thickness = [10] * skeleton.num_limbs
                colors_tmp = skeleton.colors.copy()
                for l in range(skeleton.num_limbs):
                    if not skeleton.camera_see_limb(camNet[c].cam_id, l):
                        thickness[l] = 5
                        colors_tmp[l] = (125, 125, 125)
                plt_list[i].set_data(
                    camNet[c].plot_2d(img=np.ones((480, 960, 3), dtype=np.uint8), img_id=img_id, pts=pts2d,
                                      flip_points=False, colors=colors_tmp, thickness=thickness,
                                      draw_joints=draw_joints, zorder=zorder))
            elif r == 4:  # 3d pose
                ax_3d = plt_list[i]
                points3d = pts_t[img_id, :, :]

                ang = -90 - (camNet[c].rvec[1] * 57.2)
                period = 180.0  # frames
                extend = -20
                # rotate cameras
                ang_move = np.sin(2 * np.pi * ((img_id) / period)) * extend
                ang += ang_move

                ax_3d.elev = 20
                draw_joints = [j for j in range(skeleton.num_joints) if (
                            skeleton.is_tracked_point(j, skeleton.Tracked.COXA_FEMUR) or skeleton.is_tracked_point(j,
                                                                                                                   skeleton.Tracked.FEMUR_TIBIA) or skeleton.is_tracked_point(
                        j, skeleton.Tracked.TIBIA_TARSUS) or skeleton.is_tracked_point(j, skeleton.Tracked.TARSUS_TIP))]

                colors_tmp = skeleton.colors.copy()
                thickness = [config["line_thickness"]] * skeleton.num_limbs

                plot_drosophila_3d(ax_3d=ax_3d, points3d=points3d, cam_id=c, ang=ang, draw_joints=draw_joints,
                                   zorder=zorder, colors=colors_tmp, thickness=thickness, lim=2)

    # plt.savefig(os.path.join(output_folder, f'row5_{img_id}.png'), bbox_inches = 'tight',
    #  pad_inches = 0)
    for c in range(num_cols):
        if num_rows >= 4:
            r = 4
            i = r * num_cols + c
            ax_3d = plt_list[i]
            ax_3d.cla()





























