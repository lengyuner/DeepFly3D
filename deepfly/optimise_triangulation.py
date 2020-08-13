import anipose_cameras
import CameraNetwork

def optimise_3d(cameras, points2d, points3d):
	'''
	Applys spatio-temporal filtering described in anipose

	Returns optimised 3d points
	'''
	#Setup the anipose data structures appropriately
	assert len(cameras) == 3
	#assert constraints about points2d
	#assert constraints about points3d

	#TODO check the format of the 2d and 3d points
	#TODO make sure theres nothing going wrong with the cordinate system of the 3d points
	#TODO look at anipose/triangulate.triangulate, determine what things done before and after the cgroup.optim_points call are important

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


	optimised_points3d = cgroup.optim_points(points2d, points3d) #TODO add in other paramaters to the function call

	#TODO reshape optimised_points3d


	return optimised_points3d

