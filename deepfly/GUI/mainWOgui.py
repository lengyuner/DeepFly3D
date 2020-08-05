import ast
import pickle
import sys
from sklearn.neighbors import NearestNeighbors
from deepfly.pose2d import ArgParse
from deepfly.pose2d.drosophila import main as pose2d_main
from deepfly.pose3d.procrustes.procrustes import procrustes_seperate

from .CameraNetwork import CameraNetwork
from .DB import PoseDB
from .State import State, View, Mode

from .util.main_util import button_set_width
from .util.optim_util import energy_drosoph
from .util.os_util import *

from deepfly.CLI.core_api import known_users
import re


class DrosophAnnot():
    def __init__(self, folder, n=None):
        self.chosen_points = []
        self.folder = folder
        max_num_images = n

        if self.folder.endswith("/"):
            self.folder = self.folder[:-1]
        self.folder_output = os.path.join(self.folder, 'df3d/')
        if not os.path.exists(self.folder_output):
            print(self.folder_output)
            os.makedirs(self.folder_output)
        self.state = State(self.folder)
        self.state.mode = Mode.CORRECTION #change set correction mode to do the beleif propergation
        self.state.max_num_images = max_num_images

        self.state.db = PoseDB(self.folder_output)

        # try to automatically set the camera order
        self.cidread2cid, self.cid2cidread = read_camera_order(self.folder_output)
        camera_order = None
        for regex, ordering in known_users:
            if re.search(regex, self.folder):
                camera_order = ordering
                print(f"Regexp success: {regex}, {ordering}")
                break
        if camera_order is not None:
            write_camera_order(self.folder_output, np.array(camera_order))
            self.cidread2cid, self.cid2cidread = read_camera_order(self.folder_output)
            print(self.cid2cidread)

        # find number of images in the folder
        max_img_id = get_max_img_id(self.folder)
        self.state.num_images = max_img_id + 1
        if self.state.max_num_images is not None:
            self.state.num_images = min(
                self.state.num_images, self.state.max_num_images
            )
        else:
            self.state.num_images = self.state.num_images
        print("Number of images: {}".format(self.state.num_images))
        
        self.set_cameras()

        self.image_pose_list = [
            ImagePose(self.state, self.camNetLeft[0]),
            ImagePose(self.state, self.camNetLeft[1]),
            ImagePose(self.state, self.camNetLeft[2])
        ]

        self.image_pose_list_bot = [
            ImagePose(self.state, self.camNetRight[0]),
            ImagePose(self.state, self.camNetRight[1]),
            ImagePose(self.state, self.camNetRight[2])
        ]
        # setting the initial state
        #self.set_pose(self.state.img_id)
        #change do beleif propegation on all images not just the first one
        for i in range(0, self.state.num_images):
            self.set_pose(i)
        self.set_mode(self.state.mode)

    def set_cameras(self):
        calib = read_calib(self.folder_output)
        self.camNetAll = CameraNetwork(
            image_folder=self.folder,
            output_folder=self.folder_output,
            cam_id_list=range(config["num_cameras"]),
            cid2cidread=self.cid2cidread,
            num_images=self.state.num_images,
            calibration=calib,
            num_joints=config["skeleton"].num_joints,
            heatmap_shape=config["heatmap_shape"],
        )
        self.camNetLeft = CameraNetwork(
            image_folder=self.folder,
            output_folder=self.folder_output,
            cam_id_list=config["left_cameras"],
            num_images=self.state.num_images,
            calibration=calib,
            num_joints=config["skeleton"].num_joints,
            cid2cidread=[self.cid2cidread[cid] for cid in config["left_cameras"]],
            heatmap_shape=config["heatmap_shape"],
            cam_list=[
                cam for cam in self.camNetAll if cam.cam_id in config["left_cameras"]
            ],
        )
        self.camNetRight = CameraNetwork(
            image_folder=self.folder,
            output_folder=self.folder_output,
            cam_id_list=config["right_cameras"],
            num_images=self.state.num_images,
            calibration=calib,
            num_joints=config["skeleton"].num_joints,
            cid2cidread=[self.cid2cidread[cid] for cid in config["right_cameras"]],
            heatmap_shape=config["heatmap_shape"],
            cam_list=[self.camNetAll[cam_id] for cam_id in config["right_cameras"]],
        )

        self.state.camNetLeft = self.camNetLeft
        self.state.camNetRight = self.camNetRight
        self.state.camNetAll = self.camNetAll
        self.camNetLeft.bone_param = config["bone_param"]
        self.camNetRight.bone_param = config["bone_param"]

        calib = read_calib(config["calib_fine"])
        self.camNetAll.load_network(calib)

    def pose2d_estimation(self):
        parser = ArgParse.create_parser()
        args, _ = parser.parse_known_args()
        args.checkpoint = False
        args.unlabeled = self.folder
        args.resume = config["resume"]
        args.stacks = config["num_stacks"]
        args.test_batch = config["batch_size"]
        args.img_res = [config["heatmap_shape"][0] * 4, config["heatmap_shape"][1] * 4]
        args.hm_res = config["heatmap_shape"]
        args.num_classes = config["num_predict"]

        args.max_img_id = self.state.num_images - 1
        # run the main, it will save the heatmaps and predictions in the image folder
        _, _ = pose2d_main(args)

        # makes sure cameras use the latest heatmaps and predictions
        self.set_cameras()
        self.set_mode(Mode.POSE)

        for ip in self.image_pose_list:
            ip.cam = self.camNetAll[ip.cam.cam_id]

        for ip in self.image_pose_list_bot:
            ip.cam = self.camNetAll[ip.cam.cam_id]


    def set_mode(self, mode):
        if (
            (
                mode == Mode.POSE
                and self.camNetLeft.has_pose()
                and self.camNetRight.has_pose()
            )
            or (mode == Mode.HEATMAP and self.camNetLeft.has_heatmap())
            or mode == Mode.IMAGE
            or (mode == Mode.CORRECTION and self.camNetLeft.has_pose())
        ):
            self.state.mode = mode
        else:
            print("Cannot set mode: {}".format(mode))
        if self.state.mode == Mode.CORRECTION:
            self.set_pose(self.state.img_id)

    def solve_bp(self, save_correction=False, side="left"):
        if not (
            self.state.mode == Mode.CORRECTION
            and self.state.solve_bp
            and self.camNetLeft.has_calibration()
            and self.camNetLeft.has_pose()
        ):
            print("solve BP exiting w/o run")
            return

        assert side in ["left", "right"]
        cam_net_this_side = []
        image_pose_list_this_side = []
        if side == "left":
            cam_net_this_side = self.state.camNetLeft
            image_pose_list_this_side = self.image_pose_list
        elif side == "right":
            cam_net_this_side = self.state.camNetRight
            image_pose_list_this_side = self.image_pose_list_bot

        prior = list()
        for ip in image_pose_list_this_side:
            if ip.dynamic_pose is not None:
                for (joint_id, pt2d) in ip.dynamic_pose.manual_correction_dict.items():
                    prior.append(
                        (ip.cam.cam_id, joint_id, pt2d / config["image_shape"])
                    )
        #print("Prior for BP: {}".format(prior))
        pts_bp = cam_net_this_side.solveBP( #why is it only solving BP for the left camnet??
            self.state.img_id, config["bone_param"], prior=prior
        )
        pts_bp = np.array(pts_bp)

        # set points which are not estimated by bp
        for idx, image_pose in enumerate(image_pose_list_this_side):
            pts_bp_ip = pts_bp[idx] * config["image_shape"]
            pts_bp_rep = self.state.db.read(image_pose.cam.cam_id, self.state.img_id)
            if pts_bp_rep is None:
                pts_bp_rep = image_pose.cam.points2d[self.state.img_id, :]
            else:
                pts_bp_rep *= config["image_shape"]
            pts_bp_ip[pts_bp_ip == 0] = pts_bp_rep[pts_bp_ip == 0]

            # keep track of the manually corrected points
            mcd = (
                image_pose.dynamic_pose.manual_correction_dict
                if image_pose.dynamic_pose is not None
                else None
            )
            image_pose.dynamic_pose = DynamicPose(
                pts_bp_ip, image_pose.state.img_id, joint_id=None, manual_correction=mcd
            )

        # save down corrections as training if any priors were given
#        if prior and save_correction:
        print("Saving with prior")
        for ip in image_pose_list_this_side:
            ip.save_correction()

        print("Finished Belief Propagation")

    def set_pose(self, img_id):
        print("image_id: "+str(img_id))
        self.state.img_id = img_id

        for ip in self.image_pose_list:
            ip.clear_mc()
        for ip in self.image_pose_list_bot:
            ip.clear_mc()
        if self.state.mode == Mode.CORRECTION:
            for ip in self.image_pose_list:
                pt = self.state.db.read(ip.cam.cam_id, self.state.img_id)
                modified_joints = self.state.db.read_modified_joints(
                    ip.cam.cam_id, self.state.img_id
                )
                if pt is None:
                    pt = ip.cam.points2d[self.state.img_id, :]
                else:
                    pt *= config["image_shape"]

                manual_correction = dict()
                for joint_id in modified_joints:
                    manual_correction[joint_id] = pt[joint_id]
                ip.dynamic_pose = DynamicPose(
                    pt,
                    ip.state.img_id,
                    joint_id=None,
                    manual_correction=manual_correction,
                )

            for ip in self.image_pose_list_bot:
                pt = self.state.db.read(ip.cam.cam_id, self.state.img_id)
                modified_joints = self.state.db.read_modified_joints(
                    ip.cam.cam_id, self.state.img_id
                )
                if pt is None:
                    pt = ip.cam.points2d[self.state.img_id, :]
                else:
                    pt *= config["image_shape"]

                manual_correction = dict()
                for joint_id in modified_joints:
                    manual_correction[joint_id] = pt[joint_id]
                ip.dynamic_pose = DynamicPose(
                    pt,
                    ip.state.img_id,
                    joint_id=None,
                    manual_correction=manual_correction,
                )

            if self.camNetLeft.has_calibration():
                self.solve_bp(side="left")

            if self.camNetRight.has_calibration():
                self.solve_bp(side="right")


    def set_heatmap_joint_id(self, joint_id):
        self.state.hm_joint_id = joint_id

    def calibrate_calc(self):
        from .util.main_util import calibrate_calc as ccalc

        min_img_id, max_img_id = 0, self.state.max_num_images
        ccalc(self, min_img_id, max_img_id)
        self.set_cameras()

    def save_calibration(self):
        calib_path = "{}/calib_{}.pkl".format(
            self.folder_output, self.folder.replace("/", "_")
        )
        print("Saving calibration {}".format(calib_path))
        self.camNetAll.save_network(calib_path)

    def save_pose(self):
        pts2d = np.zeros(
            (7, self.state.num_images, config["num_joints"], 2), dtype=float
        )
        # pts3d = np.zeros((self.cfg.num_images, self.cfg.num_joints, 3), dtype=float)

        for cam in self.camNetAll:
            pts2d[cam.cam_id, :] = cam.points2d.copy()

        # overwrite by manual correction
        count = 0
        for cam_id in range(config["num_cameras"]):
            for img_id in range(self.state.num_images):
                if self.state.db.has_key(cam_id, img_id):
                    pt = self.state.db.read(cam_id, img_id) * config["image_shape"]
                    pts2d[cam_id, img_id, :] = pt
                    count += 1

        if "fly" in config["name"]:
            # some post-processing for body-coxa
            for cam_id in range(len(self.camNetAll.cam_list)):
                for j in range(config["skeleton"].num_joints):
                    if config["skeleton"].camera_see_joint(cam_id, j) and config[
                        "skeleton"
                    ].is_tracked_point(j, config["skeleton"].Tracked.BODY_COXA):
                        pts2d[cam_id, :, j, 0] = np.median(pts2d[cam_id, :, j, 0])
                        pts2d[cam_id, :, j, 1] = np.median(pts2d[cam_id, :, j, 1])

        dict_merge = self.camNetAll.save_network(path=None)
        dict_merge["points2d"] = pts2d

        # take a copy of the current points2d
        pts2d_orig = np.zeros(
            (7, self.state.num_images, config["num_joints"], 2), dtype=float
        )
        for cam_id in range(config["num_cameras"]):
            pts2d_orig[cam_id, :] = self.camNetAll[cam_id].points2d.copy()

        # ugly hack to temporarly incorporate manual corrections
        c = 0
        for cam_id in range(config["num_cameras"]):
            for img_id in range(self.state.num_images):
                if self.state.db.has_key(cam_id, img_id):
                    pt = self.state.db.read(cam_id, img_id) * config["image_shape"]
                    self.camNetAll[cam_id].points2d[img_id, :] = pt
                    c += 1
        print("Replaced points2d with {} manual correction".format(count))

        # do the triangulation if we have the calibration
        if self.camNetLeft.has_calibration() and self.camNetLeft.has_pose():
            self.camNetAll.triangulate()
            pts3d = self.camNetAll.points3d_m

            dict_merge["points3d"] = pts3d
        # apply procrustes
        if config["procrustes_apply"]:
            print("Applying Procrustes on 3D Points")
            dict_merge["points3d"] = procrustes_seperate(dict_merge["points3d"])

        # put old values back
        for cam_id in range(config["num_cameras"]):
            self.camNetAll[cam_id].points2d = pts2d_orig[cam_id, :].copy()

        pickle.dump(
            dict_merge,
            open(
                os.path.join(
                    self.folder_output,
                    "pose_result_{}.pkl".format(self.folder.replace("/", "_")),
                ),
                "wb",
            ),
        )
        print(
            "Saved the pose at: {}".format(
                os.path.join(
                    self.folder_output,
                    "pose_result_{}.pkl".format(self.folder.replace("/", "_")),
                )
            )
        )


class DynamicPose:
    def __init__(self, points2d, img_id, joint_id, manual_correction=None):
        self.points2d = points2d
        self.img_id = img_id
        self.joint_id = joint_id
        self.manual_correction_dict = manual_correction
        if manual_correction is None:
            self.manual_correction_dict = dict()

    def set_joint(self, joint_id, pt2d):
        assert pt2d.shape[0] == 2
        self.points2d[joint_id] = pt2d
        self.manual_correction_dict[joint_id] = pt2d


class ImagePose():
    def __init__(self, config, cam):
        self.state = config
        self.cam = cam

        self.dynamic_pose = None

    def clear_mc(self):
        self.dynamic_pose = None


    def save_correction(self, thr=30):
        points2d_prediction = self.cam.get_points2d(self.state.img_id)
        points2d_correction = self.dynamic_pose.points2d

        err = np.abs(points2d_correction - points2d_prediction)
        check_joint_id_list = [
            j
            for j in range(config["num_joints"])
            if (j not in config["skeleton"].ignore_joint_id)
            and config["skeleton"].camera_see_joint(self.cam.cam_id, j)
        ]

        for j in check_joint_id_list:
            if np.any(err[j] > thr):
                err_max = np.max(err[check_joint_id_list])
                joint_id, ax = np.where(err == err_max)

                print(
                    "Saving camera {} with l1 {} on joint {}".format(
                        self.cam.cam_id, err_max, joint_id
                    )
                )
                # make sure we are not saving a points that cannot be seen from the camera
                unseen_joints = [
                    j
                    for j in range(config["skeleton"].num_joints)
                    if not config["skeleton"].camera_see_joint(self.cam.cam_id, j)
                ]
                points2d_correction[unseen_joints, :] = 0.0
                self.state.db.write(
                    points2d_correction / config["image_shape"],
                    self.cam.cam_id,
                    self.state.img_id,
                    train=True,
                    modified_joints=list(
                        self.dynamic_pose.manual_correction_dict.keys()
                    ),
                )

                return True

        return False


