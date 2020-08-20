import deepfly.anipose_cameras as anipose_cameras
import numpy as np
import pdb

def reshape_2d(cameras):
	'''
	Reshapes the 2d points from 'cameras' into the format needed by the anipose triangulation
	Returns a CxNxJx2 array

	C: number of camera
	N: number of frames
	J: number of joints
	K: number of constraints
	'''

	shape_2d = cameras[0].points2d.shape
	out = np.zeros([len(cameras), shape_2d[0], shape_2d[1], 2]) # 3/6, 1400, 38, 2
	for i, camera in enumerate(cameras, start=0):
		out[i, :, :, :] = camera.points2d.copy()

	out[:, :, :, 0] /= 960 # potentially need to normalise the data to between 0 and 1
	out[:, :, :, 1] /= 480

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
	del cam_list[3]
	cgroup = make_camera_group(cam_list)

	shape = cam_list[0].points2d.shape
	pts3d = np.empty([shape[0], shape[1], 3])
	for i in range(0, shape[0]): # 0-1399
		points = np.empty([len(cam_list), shape[1], 2])
		for j, cam in enumerate(cam_list, start=0):
			points[j, :, :] = cam.points2d[i, :, :] / [960, 480]

		pts3d[i, :, :] = cgroup.triangulate(points)

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
	#assert len(cameras) == 3 || 6
	#assert constraints about points2d
	#assert constraints about points3d

	#TODO check the format of the 2d and 3d points
	#TODO make sure theres nothing going wrong with the cordinate system of the 3d points
	#TODO look at anipose/triangulate.triangulate, determine what things done before and after the cgroup.optim_points call are important

	cgroup = make_camera_group(cameras)

	constraints = get_constraints()
	
	#TODO add in other paramaters to the function call
	optimised_points3d = cgroup.optim_points(points2d, points3d, constraints=constraints, verbose=True)

	return optimised_points3d


