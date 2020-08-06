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

class Annotator(QWidget):

	leg_colours = dict(front_right="#FF0000", mid_right="#0000FF", back_right="#00FF00", front_left="#FFFF00", mid_left="#FF00FF", back_left="#00FFFF")
	offsets = dict(body_coxa=0, coxa_femur=1, femur_tibia=2, tibia_tarsus=3, tarsus_tip=4, front_right=0, mid_right=5, back_right=10, front_left=19, mid_left=24, back_left=29)
	legs_list = ['front_right', 'mid_right', 'back_right', 'front_left', 'mid_left', 'back_left']
	joints_list = ['body_coxa', 'coxa_femur', 'femur_tibia', 'tibia_tarsus', 'tarsus_tip']
	limbs_list = ['coxa', 'femur', 'tibia', 'tarsus']

	def __init__(self, folder, fout):
		super().__init__()
		self.folder = folder
		self.fout = fout
		self.frame = 0
		self.camera = 0
		self.points_visibility = True

		#self.preds = np.load(glob.glob(os.path.join(self.folder, "df3d", "pose_result*"))[0], allow_pickle=True)['points2d']
		self.preds = np.load(glob.glob(os.path.join(self.folder, "df3d", "preds*"))[0], allow_pickle=True)
		self.preds[:,:,:,0] *= 960
		self.preds[:,:,:,1] *= 480
		self.pose_corr = np.load(glob.glob(os.path.join(self.folder, "df3d", "pose_corr*"))[0], allow_pickle=True)

		for i in range(0, 7):
			for j in self.pose_corr[i]:
				self.pose_corr[i][j] = self.reslice_pose_corr_to_shape(self.pose_corr[i][j], i)


		self.setGeometry(50, 50, 960, 600)
		self.bpcheckbox = self.set_bp_checkbox()

		self.this_frame_data = self.get_this_frame_pose_corr_or_pred()

		self.set_radio_buttons()
		self.set_image()
		self.frame_select = self.set_frame_select()

		self.moving_joint = -1

		self.update()
		self.show()

	def determine_same_leg(self, j1, j2):
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

	def reslice_pose_corr_to_shape(self, pose_corr_slice, camera):
		assert camera in [0,1,2,4,5,6]
		pose_corr_slice[:,0] *= 960
		pose_corr_slice[:,1] *= 480
		if camera < 3:
			return pose_corr_slice[:19, :]
		elif camera > 3:
			return pose_corr_slice[19:, :]

	def joint_number_to_colour(self, jid):
		#leg_colours = dict(front_right="#FF0000", mid_right="#0000FF", back_right="#00FF00", front_left="#FFFF00", mid_left="#FF00FF", back_left="#00FFFF")
		if jid >= 16 and jid <= 18:
			return "#554400" #TODO find correct colour for stripes

		if self.camera < 3:
			if jid >= 0 and jid <= 4:
				return "#FF0000"
			elif jid >= 5 and jid <= 9:
				return "#0000FF"
			elif jid >= 10 and jid <= 14:
				return "#00FF00"
			elif jid == 15:
				return "#222222" #TODO find correct colour for antenna
		elif self.camera > 3:
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
			self.camera = button.camera
		self.set_image()
		self.update()

	def frame_select_action(self):
		box = self.sender()
		try:
			self.frame = int(box.text())
		except:
			print("cannot convert %s to int"%(box.text()), file=sys.stderr)
		box.clearFocus()
		self.set_image()
		self.update()
		
	def get_this_frame_pose_corr_or_pred(self):
		if self.bpcheckbox.isChecked():
			try:
				return self.pose_corr[self.camera][self.frame]
			except KeyError:
				pass
		assert self.camera in [0,1,2,4,5,6]
		if self.camera < 3:
			#return self.preds[self.camera, self.frame, :19, :]
			return self.preds[self.camera, self.frame, :, :]
		elif self.camera > 3:
			return self.preds[self.camera, self.frame, :, :]
			#return self.preds[self.camera, self.frame, 19:, :]

	def set_image(self):
		try:
			self.image = QImage(self.camera_and_frame_to_fname(self.camera, self.frame))
		except:
			print("I/O error, likely frame number is too high", file=sys.stderr)
		self.this_frame_data = self.get_this_frame_pose_corr_or_pred()

	def camera_and_frame_to_fname(self, c, f):
		return os.path.join(self.folder, "camera_%d_img_%06d.jpg"%(c,f))

	def draw_points(self, painter):
		for jid, point in enumerate(self.this_frame_data, start=0):
			colour = QColor(self.joint_number_to_colour(jid))
			painter.setPen(QPen(colour, 1))
			painter.setBrush(QBrush(colour, Qt.SolidPattern))

			painter.drawEllipse(QPoint(round(point[0]), round(point[1])), 5, 5)
			if self.determine_same_leg(jid, jid+1):
				painter.drawLine(round(point[0]), round(point[1]), round(self.this_frame_data[jid+1][0]), round(self.this_frame_data[jid+1][1]))

	def next_frame(self):
		self.frame += 1
		self.set_image()
		self.frame_select.setText(str(self.frame))
		self.update()

	def previous_frame(self):
		self.frame -= 1
		self.set_image()
		self.frame_select.setText(str(self.frame))
		self.update()

	def write_frame(self):
		'''	Order in the self.this_frame_data array is different from the order in the csv file
		'''
		def make_string(d, start, stop):
			pts = []
			for i in range(start, stop):
				pts.append("%03d,%03d"%(d[i,0],d[i,1]))
			return ','.join(pts)

		d = self.this_frame_data
		fpath = self.camera_and_frame_to_fname(self.camera, self.frame)
		antenna = make_string(d, 15, 16)
		stripe = make_string(d, 16, 19)
		front_leg = make_string(d, 0, 5)
		mid_leg = make_string(d, 5, 10)
		rear_leg = make_string(d, 10, 15)

		full_string = ','.join([fpath, antenna, stripe, front_leg, mid_leg, rear_leg])
		print(full_string, file=self.fout)


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


app = QApplication(sys.argv)
fout = sys.stdout
try:
	fout = sys.argv[2]
	fout = open(fout, "a")
except:
	pass

x = Annotator(sys.argv[1], fout)
app.exec_()







