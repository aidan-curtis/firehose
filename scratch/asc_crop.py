from PIL import Image
import numpy as np
import os

# Set image fileame
from_folder = "./data/dogrib"

# Set the start x/y coordinates and size
to_folder = "./data/dogrib_c1"
start_xy = (0,0)
sizes = (100, 100)
use_const = None

# to_folder = "./data/dogrib_c2"
# start_xy = (100, 0)
# sizes = (100, 100)
# use_const = None

# to_folder = "./data/dogrib_c3"
# start_xy = (0, 100)
# sizes = (100, 100)
# use_const = None

# to_folder = "./data/mit_m"
# start_xy = (0, 0)
# sizes = (100, 100)
# use_const = 2

# to_folder = "./data/mit_i"
# start_xy = (0, 0)
# sizes = (100, 100)
# use_const = 2

to_folder = "./data/mit_t"
start_xy = (0, 0)
sizes = (100, 100)
use_const = 2

grid_names = ["cur.asc", "elevation.asc", "Forest.asc", "saz.asc", "slope.asc"]

for grid_name in grid_names:

	# Open target asc file
	input_f = open(os.path.join(from_folder, grid_name), "r")
	output_f = open(os.path.join(to_folder, grid_name), "w")

	# Write header stuff
	output_f.write("ncols {}\n".format(sizes[0]))
	output_f.write("nrows {}\n".format(sizes[1]))
	output_f.write("xllcorner 457900\n") # from dogrib
	output_f.write("yllcorner 5716800\n") # from dogrib
	output_f.write("cellsize 100\n") # from dogrib
	output_f.write("NODATA_value -9999\n") # from dogrib

	# Read data from input asc
	input_lines = input_f.readlines()

	# Strips the newline character
	all_line_vals = []
	for line in input_lines[6:]:
		line_vals = line.strip().split(" ")
		all_line_vals.append(line_vals)

	# Write data
	for x in range(start_xy[1], start_xy[1]+sizes[1]):
		if(grid_name == "Forest.asc" and use_const is not None):
			output_f.write(" ".join([str(use_const) for _ in range(sizes[0])])+"\n")		
		else:
			output_f.write(" ".join(all_line_vals[x][start_xy[0]:start_xy[0]+sizes[0]])+"\n")

	output_f.close()