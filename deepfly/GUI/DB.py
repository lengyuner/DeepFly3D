import glob
import os
import pickle

import numpy as np

from .Config import config


class PoseDB:
    def __init__(self, folder, meta=None):
        self.folder = folder

        self.db_path_list = glob.glob(os.path.join(self.folder, "pose_corr*.pkl"))
        self.last_write_image_id = 0
        if len(self.db_path_list) != 0:
            self.db_path = self.db_path_list[0]
            self.db = pickle.load(open(self.db_path, "rb"))
            if "train" not in self.db:
                self.db["train"] = {i: dict() for i in range(config["num_cameras"])}
            if "modified" not in self.db:
                self.db["modified"] = {i: dict() for i in range(config["num_cameras"])}
        else:
            self.db_path = os.path.join(
                # self.folder, "pose_corr_{}.pkl".format(self.folder.replace("/", "-"))
                self.folder, "pose_corr_{}.pkl".format('data_test')
            )
            self.db = {i: dict() for i in range(config["num_cameras"])}
            self.db["folder"] = self.folder
            self.db["meta"] = meta
            self.db["train"] = {i: dict() for i in range(config["num_cameras"])}
            self.db["modified"] = {i: dict() for i in range(config["num_cameras"])}

            self.dump()

    def read(self, cam_id, img_id):
        if img_id in self.db[cam_id]:
            return np.array(self.db[cam_id][img_id])
        else:
            return None

    def read_modified_joints(self, cam_id, img_id):
        if img_id in self.db["modified"][cam_id]:
            return np.array(self.db["modified"][cam_id][img_id])
        else:
            return []

    def write(self, pts, cam_id, img_id, train=False, modified_joints=None):
        assert pts.shape[0] == config["skeleton"].num_joints and pts.shape[1] == 2
        assert modified_joints is not None

        print("Writing {} {}".format(cam_id, img_id))
        self.db[cam_id][img_id] = pts

        if "train" not in self.db:
            self.db["train"] = {i: dict() for i in range(7)}
        if "modified" not in self.db:
            self.db["modified"] = {i: dict() for i in range(7)}

        self.db["train"][cam_id][img_id] = train
        self.db["modified"][cam_id][img_id] = modified_joints

        # too slow?
        self.dump()
        self.last_write_image_id = img_id

    def dump(self):
        print('self.db_path', '\n', self.db_path, '$' * 100) # TODO(JZ)
        with open(self.db_path, "wb") as outfile:
            pickle.dump(self.db, outfile)
        outfile.close() # TODO(JZ)

    def has_key(self, cam_id, img_id):
        return img_id in self.db[cam_id]
