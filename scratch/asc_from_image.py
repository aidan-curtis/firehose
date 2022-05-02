from PIL import Image
import numpy as np

# Set image fileame
letter = "T"
image_fn = "./scratch/images/{}.png".format(letter)
output_fn = "./scratch/images/{}.asc".format(letter)

# Load image into numpy
image = np.array(Image.open(image_fn))

# Open target asc file
output_f = open(output_fn, "w")

# Write header stuff
output_f.write("ncols {}\n".format(image.shape[0]))
output_f.write("nrows {}\n".format(image.shape[1]))
output_f.write("xllcorner 457900\n") # from dogrib
output_f.write("yllcorner 5716800\n") # from dogrib
output_f.write("cellsize 100\n") # from dogrib
output_f.write("NODATA_value -9999\n") # from dogrib

def from_color(color):
	print(color)
	return str(int(color<200))
		
# Write data
for x in range(image.shape[0]):
	# Test blue pixels
	output_f.write(" ".join([from_color(k) for k in list(image[x, :, 1])])+"\n")

output_f.close()