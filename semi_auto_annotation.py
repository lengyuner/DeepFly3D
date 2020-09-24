#!/bin/python
'''
	Programme to do semi-automatic annotation of df3d images
	deepfly3d should already be run on the target

	opens by displaying camera 0 and frame 0
	displays the image, and the points (with lines) that deepfly3d predicted
	click and drag the points to change

	user can select different cammera with radio buttons or change frame

	USAGE: python semi_automatic_annotation.py path/to/images/ [save_file.csv]

	user clicks and drags points to move them
	user presses 'w' to write the new locations to 
		stdout if there is not 2nd command line argument
		the file pointed to by the 2nd command line argument
		only writes data for the current frame
	user presses 'q' to quit
	user presses 'e' to hide joints and lines
	user presses 'n' to move to next frame
	user presses 'p' to move to previous frame
'''
import sys
import os
import stat
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import glob
import numpy as np
from abc import ABC, abstractmethod
import pdb

def np_array_to_csv_string(d):
	def make_string(d, start, stop):
		pts = []
		for i in range(start, stop):
			pts.append("%03d,%03d"%(d[i,0],d[i,1]))
		return ','.join(pts)

	antenna = make_string(d, 15, 16)
	stripe = make_string(d, 16, 19)
	front_leg = make_string(d, 0, 5)
	mid_leg = make_string(d, 5, 10)
	rear_leg = make_string(d, 10, 15)

	full_string = ','.join([antenna, stripe, front_leg, mid_leg, rear_leg])
	return full_string

class Dataloader(ABC):
	data = []
	f_in = ""
	camera = 0
	frame = 0

	@abstractmethod
	def __init__(self, f_in, f_out):
		pass
	@abstractmethod
	def get_image(self):
		pass
	@abstractmethod
	def get_data(self):
		pass
	@abstractmethod
	def next_frame(self):
		pass
	@abstractmethod
	def prev_frame(self):
		pass
	@abstractmethod
	def write(self, string):
		"""	Writes the current frame's data to self.f_out
		"""
		pass
	@abstractmethod
	def set_frame(self, frame):
		"""	Sets the frame to 'frame'
			returns True if successful, false otherwise
		"""
		pass
	@abstractmethod
	def set_camera(self, camera):
		"""	Sets the camera to 'camera'
			returns True if successful, false otherwise
		"""
		pass

class DF3DFolderDataloader(Dataloader):
	"""	Loads the frames from a video analysed using df3d
	"""

	def __init__(self, folder, f_out):
		self.f_in = folder
		self.f_out = f_out
		self.camera = 0
		self.frame = 0

		self.preds = np.load(glob.glob(os.path.join(self.f_in, "df3d", "preds*"))[0], allow_pickle=True)
		self.preds[:,:,:,0] *= 960
		self.preds[:,:,:,1] *= 480
		self.pose_corr = np.load(glob.glob(os.path.join(self.f_in, "df3d", "pose_corr*"))[0], allow_pickle=True)

		for i in range(0, 7):
			for j in self.pose_corr[i]:
				self.pose_corr[i][j] = self.reslice_pose_corr_to_shape(self.pose_corr[i][j], i)

		self.correct_preds_stripe()

	def get_image(self):
		try:
			return QImage(self.camera_and_frame_to_fname(self.camera, self.frame))
		except:
			pdb.set_trace()
			print("I/O error, likely frame number is too high", file=sys.stderr)
			return None

	def get_data(self, belief_prop=False):
		if belief_prop:
			try:
				return self.pose_corr[self.camera][self.frame]
			except KeyError:
				pass
		assert self.camera in [0,1,2,4,5,6]
		if self.camera < 3:
			return self.preds[self.camera, self.frame, :, :]
		elif self.camera > 3:
			return self.preds[self.camera, self.frame, :, :]

	def next_frame(self):
		self.frame += 1

	def prev_frame(self):
		self.frame -= 1

	def write(self, d):
		"""	Writes the current frame's data to self.f_out
			Order in the data array is different from the order in the csv file

			d: array of data for this frame
		"""
		fpath = self.camera_and_frame_to_fname(self.camera, self.frame)
		data_string = np_array_to_csv_string(d)
		print(fpath+','+data_string, file=self.f_out)

	def set_frame(self, frame):
		"""	Sets the frame to 'frame'
			returns True if successful, false otherwise
		"""
		self.frame = frame
		return True

	def set_camera(self, camera):
		"""	Sets the camera to 'camera'
			returns True if successful, false otherwise
		"""
		self.camera = camera
		return True

	def correct_preds_stripe(self):
		self.preds[2, :, 16:19, :] = 0
		self.preds[4, :, 16:19, :] = 0

	def reslice_pose_corr_to_shape(self, pose_corr_slice, camera):
		assert camera in [0,1,2,4,5,6]
		pose_corr_slice[:,0] *= 960
		pose_corr_slice[:,1] *= 480
		if camera < 3:
			return pose_corr_slice[:19, :]
		elif camera > 3:
			return pose_corr_slice[19:, :]

	def camera_and_frame_to_fname(self, c, f):
		return os.path.join(self.f_in, "camera_%d_img_%06d.jpg"%(c,f))

class CSVDataloader(Dataloader):
	data = []
	f_in = ""
	camera = 0 # camera has no meaning in this dataloader
	frame = 0 # in this dataloader, frame has a different meaning: it refers to the line of the csv file being parsed

	def __init__(self, f_in, f_out):
		'''	f_in: path to csv file containing annotations
			f_out: path to 
		'''
		self.f_in = f_in
		self.f_out = f_out
		self.raw_csv = []
		with open(self.f_in) as csv_file:
			for line in csv_file:
				self.raw_csv.append(line.strip().split(','))
		

	def get_image(self):
		csv_entry = self.raw_csv[self.frame]
		self.camera = self.duel_path_to_cam_num(self.raw_csv[self.frame][0])
		#print("camera:%d"%(self.camera))
		#print("fpath:%s"%(self.raw_csv[self.frame][0]))
		return QImage(csv_entry[0])

	def duel_path_to_cam_num(self, path):
		'''	Works for folder1/folder2/folder3/camera_n_img_xxxxxx.jpg
			and folder1/folder2__folder3__folder4__camera_n_img_xxxxxx.jpg
		'''
		basename = os.path.basename(path)
		img_part = basename.split("__")[-1]
		return int(img_part[7])

	def get_data(self, belief_prop=False):
		csv_entry = self.raw_csv[self.frame]
		self.camera = self.duel_path_to_cam_num(self.raw_csv[self.frame][0])
		return self.csv_to_np_array(csv_entry)

	def next_frame(self):
		self.frame += 1

	def prev_frame(self):
		self.frame -= 1

	def write(self, d):
		"""	Writes the current frame's data to self.f_out
		"""
		print(self.raw_csv[self.frame][0] + ',' + np_array_to_csv_string(d), file=self.f_out)

	def set_frame(self, frame):
		"""	Sets the frame to 'frame'
			returns True if successful, false otherwise
		"""
		self.frame = frame
		return True

	def set_camera(self, camera):
		"""	Sets the camera to 'camera'
			returns True if successful, false otherwise
		"""
		self.frame = frame
		return True

	def csv_to_np_array(self, line):
		"""	line: data from the csv file split on commas eg. ['fpath',0,0,1,2,3,4,5,...]
		"""
		ANTENNA = 1
		BACK = ANTENNA + 2*1
		FRONT_LEG = BACK + 2*3
		MID_LEG = FRONT_LEG + 2*5
		BACK_LEG = MID_LEG + 2*5
		out = np.zeros([19,2])

		out[15,:] = [float(line[ANTENNA]), float(line[ANTENNA + 1])]

		out[0 ,:] = [float(line[FRONT_LEG]), float(line[FRONT_LEG + 1])]
		out[1 ,:] = [float(line[FRONT_LEG + 2*1]), float(line[FRONT_LEG + 2*1 + 1])]
		out[2 ,:] = [float(line[FRONT_LEG + 2*2]), float(line[FRONT_LEG + 2*2 + 1])]
		out[3 ,:] = [float(line[FRONT_LEG + 2*3]), float(line[FRONT_LEG + 2*3 + 1])]
		out[4 ,:] = [float(line[FRONT_LEG + 2*4]), float(line[FRONT_LEG + 2*4 + 1])]

		out[5 ,:] = [float(line[MID_LEG]), float(line[MID_LEG + 1])]
		out[6 ,:] = [float(line[MID_LEG + 2*1]), float(line[MID_LEG + 2*1 + 1])]
		out[7 ,:] = [float(line[MID_LEG + 2*2]), float(line[MID_LEG + 2*2 + 1])]
		out[8 ,:] = [float(line[MID_LEG + 2*3]), float(line[MID_LEG + 2*3 + 1])]
		out[9 ,:] = [float(line[MID_LEG + 2*4]), float(line[MID_LEG + 2*4 + 1])]

		out[10,:] = [float(line[BACK_LEG]), float(line[BACK_LEG + 1])]
		out[11,:] = [float(line[BACK_LEG + 2*1]), float(line[BACK_LEG + 2*1 + 1])]
		out[12,:] = [float(line[BACK_LEG + 2*2]), float(line[BACK_LEG + 2*2 + 1])]
		out[13,:] = [float(line[BACK_LEG + 2*3]), float(line[BACK_LEG + 2*3 + 1])]
		out[14,:] = [float(line[BACK_LEG + 2*4]), float(line[BACK_LEG + 2*4 + 1])]

		out[16,:] = [float(line[BACK]), float(line[BACK + 1])]
		out[17,:] = [float(line[BACK + 2*1]), float(line[BACK + 2*1 + 1])]
		out[18,:] = [float(line[BACK + 2*2]), float(line[BACK + 2*2 + 1])]


		#out[out < 10] = 0
		return out


class Annotator(QWidget):

	leg_colours = dict(front_right="#FF0000", mid_right="#0000FF", back_right="#00FF00", front_left="#FFFF00", mid_left="#FF00FF", back_left="#00FFFF")
	offsets = dict(body_coxa=0, coxa_femur=1, femur_tibia=2, tibia_tarsus=3, tarsus_tip=4, front_right=0, mid_right=5, back_right=10, front_left=19, mid_left=24, back_left=29)
	legs_list = ['front_right', 'mid_right', 'back_right', 'front_left', 'mid_left', 'back_left']
	joints_list = ['body_coxa', 'coxa_femur', 'femur_tibia', 'tibia_tarsus', 'tarsus_tip']
	limbs_list = ['coxa', 'femur', 'tibia', 'tarsus']

	def __init__(self, folder, fout, dataloader):
		super().__init__()

		self.dataloader = dataloader

		self.setGeometry(50, 50, 960, 600)
		self.bpcheckbox = self.set_bp_checkbox()

		self.points_visibility = True
		self.this_frame_data = self.dataloader.get_data(belief_prop=self.bpcheckbox.isChecked())

		self.set_radio_buttons()
		self.set_image()
		self.frame_select = self.set_frame_select()

		self.moving_joint = -1

		self.update()
		self.show()

	@staticmethod
	def determine_same_leg(j1, j2):
		'''	Determines if joint ids j1 and j2 are part of the same leg/structure
			j2 = j1 + 1
		'''
		assert j2 == j1 + 1 #not implemented otherwise

		if j1 == 15:
			return False
		elif j1 >= 0 and j1 <= 3:
			return True
		elif j1 >= 5 and j1 <= 8:
			return True
		elif j1 >= 10 and j1 <= 13:
			return True
		elif j1 >= 16 and j1 <= 17:
			return True

		return False

	def joint_number_to_colour(self, jid):
		#leg_colours = dict(front_right="#FF0000", mid_right="#0000FF", back_right="#00FF00", front_left="#FFFF00", mid_left="#FF00FF", back_left="#00FFFF")
		if jid >= 16 and jid <= 18:
			return "#554400" #TODO find correct colour for stripes

		if self.dataloader.camera < 3:
			if jid >= 0 and jid <= 4:
				return "#FF0000"
			elif jid >= 5 and jid <= 9:
				return "#0000FF"
			elif jid >= 10 and jid <= 14:
				return "#00FF00"
			elif jid == 15:
				return "#222222" #TODO find correct colour for antenna
		elif self.dataloader.camera > 3:
			if jid >= 0 and jid <= 4:
				return "#FFFF00"
			elif jid >= 5 and jid <= 9:
				return "#FF00FF"
			elif jid >= 10 and jid <= 14:
				return "#00FFFF"
			elif jid == 15:
				return "#222222" #TODO find correct colour for antenna

		print("invalid joint id", file=sys.stderr)
		exit(12)

	def set_frame_select(self):
		box = QLineEdit("0", self)
		box.setGeometry(100, 540, 100, 30)
		box.returnPressed.connect(self.frame_select_action)
		lab = QLabel("Frame Number", self)
		lab.setGeometry(210, 540, 100, 30)
		return box

	def bp_box_toggle(self):
		self.this_frame_data = self.get_this_frame_pose_corr_or_pred()
		self.update()

	def set_bp_checkbox(self):
		box = QCheckBox("Show BP", self)
		box.setGeometry(400, 540, 100, 30)
		box.stateChanged.connect(self.bp_box_toggle)
		return box

	def radio_button_clicked(self):
		button = self.sender()
		if button.isChecked():
			self.dataloader.camera = button.camera
		self.set_image()
		self.update()

	def frame_select_action(self):
		box = self.sender()
		try:
			if not self.dataloader.set_frame(int(box.text())):
				print("cannot set frame to %s"%(box.text()), file=sys.stderr)
		except:
			print("cannot convert %s to int"%(box.text()), file=sys.stderr)
		box.clearFocus()
		self.set_image()
		self.update()
		
	def get_this_frame_pose_corr_or_pred(self):
		return self.dataloader.get_data(belief_prop=self.bpcheckbox.isChecked())

	def set_image(self):
		self.image = self.dataloader.get_image()
		self.this_frame_data = self.dataloader.get_data(belief_prop=self.bpcheckbox.isChecked())

	def draw_points(self, painter):
		for jid, point in enumerate(self.this_frame_data, start=0):
			colour = QColor(self.joint_number_to_colour(jid))
			painter.setPen(QPen(colour, 1))
			painter.setBrush(QBrush(colour, Qt.SolidPattern))

			painter.drawEllipse(QPoint(round(point[0]), round(point[1])), 5, 5)
			if self.determine_same_leg(jid, jid+1):
				painter.drawLine(round(point[0]), round(point[1]), round(self.this_frame_data[jid+1][0]), round(self.this_frame_data[jid+1][1]))

	def next_frame(self):
		self.dataloader.next_frame()
		self.set_image()
		self.frame_select.setText(str(self.dataloader.frame))
		self.update()

	def previous_frame(self):
		self.dataloader.prev_frame()
		self.set_image()
		self.frame_select.setText(str(self.dataloader.frame))
		self.update()

	def write_frame(self):
		self.dataloader.write(self.this_frame_data)

	def paintEvent(self, e):
		painter = QPainter()
		painter.begin(self)
		painter.drawImage(0,0,self.image)
		if self.points_visibility:
			self.draw_points(painter)
		painter.end()

	def toggle_points_visibility(self):
		if self.points_visibility:
			self.points_visibility = False
		else:
			self.points_visibility = True
		self.update()

	def keyPressEvent(self, e):
		key = e.key()
		if key == 81: #q
			exit(0)
		elif key == 87: #w
			self.write_frame()
		elif key == 69: #e
			self.toggle_points_visibility()
		elif key == 78: #n
			self.next_frame()
		elif key == 80: #p
			self.previous_frame()

	def mousePressEvent(self, e):
		cursor_pos = np.array([e.x(), e.y()])
		if cursor_pos[0] > 960 or cursor_pos[1] > 480:
			return

		QGraphicsView.setMouseTracking(self, True)
		# find nearest point
		min_dist = 99999
		min_dist_index = 99999
		for jid, point in enumerate(self.this_frame_data, start=0):
			dist = np.linalg.norm(cursor_pos - point)
			if dist < min_dist:
				min_dist = dist
				min_dist_index = jid
		self.this_frame_data[min_dist_index] = cursor_pos
		self.moving_joint = min_dist_index
		self.update()


	def mouseReleaseEvent(self, e):
		QGraphicsView.setMouseTracking(self, False)
		self.moving_joint = -1

	def mouseMoveEvent(self, e):
		if e.x() > 960 or e.y() > 480 or e.x() < 0 or e.y() < 0:
			return
		self.this_frame_data[self.moving_joint] = np.array([e.x(), e.y()])
		self.update()

	def set_radio_buttons(self):
		radiobutton = QRadioButton("camera_0", self)
		radiobutton.setChecked(True)
		radiobutton.camera = 0
		radiobutton.setGeometry(100,500,200,20)
		radiobutton.toggled.connect(self.radio_button_clicked)

		radiobutton = QRadioButton("camera_1", self)
		radiobutton.camera = 1
		radiobutton.setGeometry(220,500,200,20)
		radiobutton.toggled.connect(self.radio_button_clicked)

		radiobutton = QRadioButton("camera_2", self)
		radiobutton.camera = 2
		radiobutton.setGeometry(340,500,200,20)
		radiobutton.toggled.connect(self.radio_button_clicked)

		radiobutton = QRadioButton("camera_4", self)
		radiobutton.camera = 4
		radiobutton.setGeometry(460,500,200,20)
		radiobutton.toggled.connect(self.radio_button_clicked)

		radiobutton = QRadioButton("camera_5", self)
		radiobutton.camera = 5
		radiobutton.setGeometry(580,500,200,20)
		radiobutton.toggled.connect(self.radio_button_clicked)

		radiobutton = QRadioButton("camera_6", self)
		radiobutton.camera = 6
		radiobutton.setGeometry(700,500,200,20)
		radiobutton.toggled.connect(self.radio_button_clicked)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	fout = sys.stdout
	try:
		fout = sys.argv[2]
		fout = open(fout, "a")
	except:
		pass

	if sys.argv[1].endswith(".csv"):
		dataloader = CSVDataloader(sys.argv[1], fout)
	elif os.path.isdir(sys.argv[1]):
		dataloader = DF3DFolderDataloader(sys.argv[1], fout)
	else:
		print("Requires either a csv file or a df3d directory as input", file=sys.stderr)
		exit(1)

	x = Annotator(sys.argv[1], fout, dataloader)
	app.exec_()







