"""
- name: Matheus Aparecido do Carmo Alves
- usp number: 9791114
- course code: SCC0251
- year/semester: 2020/1
- github repo: https://github.com/Micanga/image_enhancement_and_filtering

- title of the assignment:
Assignment 2: Image Enhancement and Filtering
"""
#####
# IMPORTS AND CONSTANTS
#####
import imageio
import numpy

BILATERAL_FILTERING = 1
UNSHARP_MASK		= 2
VIGNETTE_FILTER		= 3

#####
# METHODS IMPLEMENTATION
#####
def euclidean_distance(x,y):
	return float(numpy.sqrt(x**2 + y**2))

def gaussian_kernel(x,omega):
	weight = (2.0*numpy.pi*(omega**2))**(-1)
	exp = -((x**2)/(2.0*(omega**2)))
	return float(weight*numpy.exp(exp))

def bilateral_filtering(input_img,n,omega_s,omega_r):
	# 1. Creating the output image
	width, height = input_img.shape
	output_img = numpy.zeros((width,height))

	# 2. Applying the bilateral filtering
	# a. computing the spatial Gaussian component
	spatial_matrix = numpy.zeros((n,n))
	center_x, center_y = int(n/2), int(n/2)

	# - calculating the components and difining the filter
	for x in range(n):
		for y in range(n):
			distance = euclidean_distance((x - center_x),(y - center_y))
			spatial_matrix[x,y] = gaussian_kernel(distance,omega_s)

	# b. applying the convolution
	for x in range(width):
		for y in range(height):
			If, Wp = 0.0, 0.0

			for n_x in range(n):
				for n_y in range(n):
					# - getting current pixel value
					Ixy = input_img[x,y]

					# - defining the neighbor pixel value
					neighbor_x = (x + (n_x - center_x))
					neighbor_y = (y + (n_y - center_y))

					Ii = input_img[neighbor_x,neighbor_y] if 0 <=  neighbor_x < width\
						and 0 <=  neighbor_y < height else 0.0

					# - calculating the new pixel value
					delta_I = float(Ii) - float(Ixy)
					gri = gaussian_kernel(delta_I, omega_r)

					wi = float(gri) * float(spatial_matrix[n_x,n_y])
					Wp = Wp + float(wi)
					If = If + float(wi * Ii)

			output_img[x,y] = int(If/Wp)

	# 3. Returning the result image
	return output_img

def unsharp_mask(input_img,c,kernel_idx):
	# 1. Creating the output image
	width, height = input_img.shape
	output_img = numpy.zeros((width,height))

	# 2. Defining the kernels
	if kernel_idx == 1:
		kernel = numpy.array([[ 0,-1, 0],\
						   [-1, 4,-1],\
						   [ 0,-1, 0]])
	else:
		kernel = numpy.array([[-1,-1,-1],\
						   [-1, 8,-1],\
						   [-1,-1,-1]])
	kernel_w, kernel_h = kernel.shape
	center_x, center_y = int(kernel_w/2), int(kernel_h/2)

	# 3. Applying the unsharp mask
	# a. convolving
	for x in range(width):
		for y in range(height):
			If = 0

			for k_x in range(kernel_w):
				for k_y in range(kernel_h):
					neighbor_x = (x + (k_x - center_x))
					neighbor_y = (y + (k_y - center_y))
					Ii = input_img[neighbor_x,neighbor_y] if 0 <=  neighbor_x < width\
						and 0 <=  neighbor_y < height else 0
					If = If + (Ii * kernel[k_x,k_y])

			output_img[x,y] = If

	# b. scaling
	min_i = numpy.min(output_img)
	max_i = numpy.max(output_img)
	for x in range(width):
		for y in range(height):
			output_img[x,y] = ((output_img[x,y]-min_i)*255/(max_i-min_i))

	# c. adding
	for x in range(width):
		for y in range(height):
			output_img[x,y] = (c*output_img[x,y]) + input_img[x,y]

	# d. scaling 
	min_i = numpy.min(output_img)
	max_i = numpy.max(output_img)
	for x in range(width):
		for y in range(height):
			output_img[x,y] = ((output_img[x,y]-min_i)*255/(max_i-min_i))

	# 4. Returning the result image
	return output_img

def vignette_filter(input_img,omega_row,omega_col):
	# 1. Creating the output image
	width, height = input_img.shape
	output_img = numpy.zeros((width,height))

	# 2. Applying the vignette filter
	# a. creating the gaussian kernels
	center_x, center_y = width/2 -1 , height/2 -1
	Wrow = numpy.array([[gaussian_kernel(r-center_x,omega_row) for r in range(width)]])
	Wcol = numpy.array([[gaussian_kernel(c-center_y,omega_col) for c in range(height)]])

	# b. calculating the main kernel
	W = numpy.matmul(Wrow.T,Wcol)

	# c. applying it the filter
	for x in range(width):
		for y in range(height):
			output_img[x,y] = float(input_img[x,y])*W[x,y]

	# d. normalising
	min_i = numpy.min(output_img)
	max_i = numpy.max(output_img)
	for x in range(width):
		for y in range(height):
			output_img[x,y] = ((output_img[x,y]-min_i)*255/(max_i-min_i))

	# 3. Returning the result image
	return output_img

def RSE(input_img,output_img):
	RSE = 0.0 
	width, height = input_img.shape
	for i in range(width):
		for j in range(height):
			RSE += (output_img[i,j] - float(input_img[i,j]))**2
	RSE = numpy.sqrt(RSE)
	return RSE

#####
# MAIN CODE
#####
# 1. Reading the inputs
# a. image
filename = str(input()).rstrip()
input_img = imageio.imread(filename)

# b. desired method
method = int(input())

# c. save parameter
save = int(input())

# 2. Starting the image processing
# a. Bilateral Filtering
if method == BILATERAL_FILTERING:
	# - reading additional inputs
	# size of filter n, parameters omega_s
	# and omega_r
	n = int(input())
	omega_s = float(input())
	omega_r = float(input())

	# - applying the bilateral filter
	output_img = bilateral_filtering(input_img,n,omega_s,omega_r)

# b. Unsharp mask using the Laplacian Filter
elif method == UNSHARP_MASK:
	# - reading additional inputs
	# constant c and kernel
	c = float(input())
	kernel_idx = int(input())

	# - applying the unsharp mask
	output_img = unsharp_mask(input_img,c,kernel_idx)

# c. Vignette Filter
elif method == VIGNETTE_FILTER:
	# - reading additional inputs
	# omega_row and omega_col
	omega_row = float(input())
	omega_col = float(input())

	# - applying the unsharp mask
	output_img = vignette_filter(input_img,omega_row,omega_col)

# d. Invalid method
else:
	print('InputError: invalid input method (integer between 1 and 3).')
	exit(1)

# 3. Saving the image (if requested)
if save == 1:
	imageio.imwrite('output_img.png',output_img.astype(numpy.uint8))

# 4. Caculating the RSE
RSE = RSE(input_img,output_img)
print('%.4f' % RSE)

# That's all folks... :}