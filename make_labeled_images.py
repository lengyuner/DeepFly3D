import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import semi_auto_annotation

def main():
	labels_file = open(sys.argv[1], "r")
	save_to = os.path.join("data", "labeled")

	for line in labels_file:
		# setup the data
		fields = line.split(',')
		fpath_and_name = fields[0]
		cid, data = parse_csv_annotations(fields)
		process_data_to_shape(data, cid)
		data = data[:19, :]
		image = plt.imread(fpath_and_name)

		#produce the graph
		plt.imshow(image)
		for i, p in enumerate(data, start=0):
			if semi_auto_annotation.Annotator.determine_same_leg(i, i+1):
				p1 = data[i,:]
				p2 = data[i+1, :]
				plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=joint_number_to_colour(i, cid), linewidth=0.65, marker='o')

		plt.savefig(os.path.join(save_to, fpath_and_name.replace("/", "-")))
		plt.close()


def parse_csv_annotations(line):
    ANTENNA = 1
    BACK = ANTENNA + 2*1
    FRONT_LEG = BACK + 2*3
    MID_LEG = FRONT_LEG + 2*5
    BACK_LEG = MID_LEG + 2*5
    cid = int(os.path.basename(line[0])[7])
    out = np.zeros([40,2])
    if cid < 3:
        out[30,:] = [float(line[ANTENNA]), float(line[ANTENNA + 1])]

        out[0,:] = [float(line[FRONT_LEG]), float(line[FRONT_LEG + 1])]
        out[1,:] = [float(line[FRONT_LEG + 2*1]), float(line[FRONT_LEG + 2*1 + 1])]
        out[2,:] = [float(line[FRONT_LEG + 2*2]), float(line[FRONT_LEG + 2*2 + 1])]
        out[3,:] = [float(line[FRONT_LEG + 2*3]), float(line[FRONT_LEG + 2*3 + 1])]
        out[4,:] = [float(line[FRONT_LEG + 2*4]), float(line[FRONT_LEG + 2*4 + 1])]

        out[5 ,:] = [float(line[MID_LEG]), float(line[MID_LEG + 1])]
        out[6 ,:] = [float(line[MID_LEG + 2*1]), float(line[MID_LEG + 2*1 + 1])]
        out[7 ,:] = [float(line[MID_LEG + 2*2]), float(line[MID_LEG + 2*2 + 1])]
        out[8 ,:] = [float(line[MID_LEG + 2*3]), float(line[MID_LEG + 2*3 + 1])]
        out[9 ,:] = [float(line[MID_LEG + 2*4]), float(line[MID_LEG + 2*4 + 1])]

        out[10 ,:] = [float(line[BACK_LEG]), float(line[BACK_LEG + 1])]
        out[11 ,:] = [float(line[BACK_LEG + 2*1]), float(line[BACK_LEG + 2*1 + 1])]
        out[12 ,:] = [float(line[BACK_LEG + 2*2]), float(line[BACK_LEG + 2*2 + 1])]
        out[13 ,:] = [float(line[BACK_LEG + 2*3]), float(line[BACK_LEG + 2*3 + 1])]
        out[14 ,:] = [float(line[BACK_LEG + 2*4]), float(line[BACK_LEG + 2*4 + 1])]

        out[31,:] = [float(line[BACK]), float(line[BACK + 1])]
        out[32,:] = [float(line[BACK + 2*1]), float(line[BACK + 2*1 + 1])]
        out[33,:] = [float(line[BACK + 2*2]), float(line[BACK + 2*2 + 1])]

    if cid > 3:
        out[35,:] = [float(line[ANTENNA]), float(line[ANTENNA + 1])]

        out[15,:] = [float(line[FRONT_LEG]), float(line[FRONT_LEG + 1])]
        out[16,:] = [float(line[FRONT_LEG + 2*1]), float(line[FRONT_LEG + 2*1 + 1])]
        out[17,:] = [float(line[FRONT_LEG + 2*2]), float(line[FRONT_LEG + 2*2 + 1])]
        out[18,:] = [float(line[FRONT_LEG + 2*3]), float(line[FRONT_LEG + 2*3 + 1])]
        out[19,:] = [float(line[FRONT_LEG + 2*4]), float(line[FRONT_LEG + 2*4 + 1])]

        out[20,:] = [float(line[MID_LEG]), float(line[MID_LEG + 1])]
        out[21,:] = [float(line[MID_LEG + 2*1]), float(line[MID_LEG + 2*1 + 1])]
        out[22,:] = [float(line[MID_LEG + 2*2]), float(line[MID_LEG + 2*2 + 1])]
        out[23,:] = [float(line[MID_LEG + 2*3]), float(line[MID_LEG + 2*3 + 1])]
        out[24,:] = [float(line[MID_LEG + 2*4]), float(line[MID_LEG + 2*4 + 1])]

        out[25,:] = [float(line[BACK_LEG]), float(line[BACK_LEG + 1])]
        out[26,:] = [float(line[BACK_LEG + 2*1]), float(line[BACK_LEG + 2*1 + 1])]
        out[27,:] = [float(line[BACK_LEG + 2*2]), float(line[BACK_LEG + 2*2 + 1])]
        out[28,:] = [float(line[BACK_LEG + 2*3]), float(line[BACK_LEG + 2*3 + 1])]
        out[29,:] = [float(line[BACK_LEG + 2*4]), float(line[BACK_LEG + 2*4 + 1])]

        out[36,:] = [float(line[BACK]), float(line[BACK + 1])]
        out[37,:] = [float(line[BACK + 2*1]), float(line[BACK + 2*1 + 1])]
        out[38,:] = [float(line[BACK + 2*2]), float(line[BACK + 2*2 + 1])]
    if cid == 3:
        pass
    out[out < 10] = 0

    return (cid, out)

def process_data_to_shape(v, cid):
	if cid > 3:  # then right camera
		v[0:15, :] = v[15:30, :]
		v[15] = v[35]  # 35 is the second antenna
		v[16:19] = v[36:39] # stripes
	elif cid < 3:
		v[15] = v[30]  # 30 is the first antenna
		v[16:19] = v[31:34] #stripes
	elif cid== 3 or cid == 7:
		pass
	else:
		raise NotImplementedError
	v[19:, :] = 0.0

def joint_number_to_colour(jid, cid):
	#leg_colours = dict(front_right="#FF0000", mid_right="#0000FF", back_right="#00FF00", front_left="#FFFF00", mid_left="#FF00FF", back_left="#00FFFF")
	if jid >= 16 and jid <= 18:
		return "#554400" #TODO find correct colour for stripes

	if cid < 3:
		if jid >= 0 and jid <= 4:
			return "#FF0000"
		elif jid >= 5 and jid <= 9:
			return "#0000FF"
		elif jid >= 10 and jid <= 14:
			return "#00FF00"
		elif jid == 15:
			return "#222222" #TODO find correct colour for antenna
	elif cid > 3:
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

if __name__ == "__main__":
	main()
