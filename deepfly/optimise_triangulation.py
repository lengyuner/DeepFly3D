import deepfly.anipose_cameras as anipose_cameras
import numpy as np
import pdb
from deepfly.Config import config

def reshape_2d(cameras):
	'''
	Reshapes the 2d points from 'cameras' into the format needed by the anipose triangulation
	Returns a CxNxJx2 array

	C: number of camera
	N: number of frames
	J: number of joints
	K: number of constraints

	0. values are replaced with np.nan
	'''

	shape_2d = cameras[0].points2d.shape
	out = np.zeros([len(cameras), shape_2d[0], shape_2d[1], 2]) # 3/6, 1400, 38, 2
	for i, camera in enumerate(cameras, start=0):
		out[i, :, :, :] = camera.points2d.copy()

	out[out == 0.] == np.nan

	return out

def make_camera_group(cameras):
	'''	Converts the deepfly3d cameras into anipose cameras
		cameras is a list of deepfly3d cameras
	'''
	#Convert deepfly3d cameras to anipose cameras
	new_cameras = []
	for cam in cameras:
		matrix = cam.intr
		dist = cam.distort
		rvec = cam.rvec
		tvec = cam.tvec
		size = (960, 480)
		extra_dist = False #???????
		#TODO check that all these values are right, and in the right format

		new_cam = anipose_cameras.Camera(matrix=matrix, dist=dist, size=size, rvec=rvec, tvec=tvec)
		new_cameras.append(new_cam)

	cgroup = anipose_cameras.CameraGroup(new_cameras, None)

	return cgroup

def anipose_3d_triangulation(cam_list):
	'''	Uses the anipose 3d simple triangulation
		camlist is a list of deepfly3d cameras.
	'''
	assert len(cam_list) == 6
	cgroup = make_camera_group(cam_list) # ordering of cameras in cgroup should be same as in deepfly
	shape = cam_list[0].points2d.shape # 1400, 38, 2
	pts2d = np.empty([6, shape[0], shape[1], 2]) # (6, 1400, 38, 2)
	for i, cam in enumerate(cam_list, start=0):
		pts2d[i, :, :, :] = cam.points2d.copy()
	pts2d[pts2d == 0.] = np.nan
	pts3d = np.empty([shape[0], shape[1], 3]) # 1400, 38, 3
	'''
	for i in range(0, shape[0]): # 0-1399
		for j in range(0, shape[1]): # 0-37
			# if camera can see joint
			pdb.set_trace()
			cam_list_iter = []
			pts2d_for_triangulate = []
			for k, camera in enumerate(cgroup.cameras, start=0):
				if not config["skeleton"].camera_see_joint(k, j):
					# remove cameras that cant see the joint
					continue
				cam_list_iter.append(camera)
			pts2d_for_triangulation = pts2d[:, i, j, :]
			pts3d[i, j, :] = cgroup.triangulate(pts2d_for_triangulation)
			reprojerr = cgroup.reprojection_error(pts3d[i,j,:], pts2d_for_triangulation, mean=True)
	'''
	pts2d_flat = pts2d.reshape(len(cam_list), -1, 2)
	pts3d_flat = cgroup.triangulate(pts2d_flat)
	reprojerr_flat = cgroup.reprojection_error(pts3d_flat, pts2d_flat, mean=True)
	pts3d = pts3d_flat.reshape(shape[0], 38, 3)
	reprojerr = reprojerr_flat.reshape(shape[0], 38)

	assert not np.any(pts3d == np.nan), "Tracking failure for at least one point"

	return pts3d

def reshape_3d(points):
	'''
	Reshapes the 3d points into the format needed by the anipose triangulation
	Returns a NxJx3 array

	N: number of frames
	J: number of joints
	'''
	# points is already be the right shape coming from deepfly3d

	return points

def get_constraints():
	'''
	Gets the constraints for the 3d triangulation optimisation
	'''
	#limb lengths on same leg should be constant
	return [[0,1], [1,2], [2,3], [3,4], [5,6], [6,7], [7,8], [8,9], [10,11], [11,12], [12,13], [13,14], [19,20], [20,21], [21,22], [22,23], [24,25], [25,26], [26,27], [27,28], [29,30], [30,31], [31,32], [32,33]]


def optimise_3d(cameras, points2d, points3d):
	'''
	Applys spatio-temporal filtering described in anipose

	Returns optimised 3d points
	'''
	#Setup the anipose data structures appropriately
	assert len(cameras) == 6
	shape_2d = points2d.shape
	shape_3d = points3d.shape
	assert len(cameras) == shape_2d[0], "number of cameras in camera list differs from number of cameras in data"
	assert shape_2d[1] == shape_3d[0], "number of 2d frames differs from number of 3d frames"
	assert shape_2d[2] == shape_3d[1], "number of joints differs between 2d and 3d data"
	assert shape_2d[3] == 2
	assert shape_3d[2] == 3
	assert not np.any(points3d == np.nan), "NaN values in 3d points: 2d tracking errors resulted in incomplete 3d triangulation"
	points2d[points2d == 0.] = np.nan

	#TODO make sure theres nothing going wrong with the cordinate system of the 3d points
	#TODO look at anipose/triangulate.triangulate, determine what things done before and after the cgroup.optim_points call are important

	cgroup = make_camera_group(cameras)

	constraints = get_constraints()
	constraints_weak = get_constraints()
	
	#TODO add in other paramaters to the function call
	optimised_points3d = cgroup.optim_points(points2d, points3d, constraints=constraints, verbose=True)

	return optimised_points3d


