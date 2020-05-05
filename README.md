# Assignment 2

In this assignment we had to implement different methods in order to enhance and/or filter images using space domain filtering (convolution). 
Read the instructions for each step in the "dip_t02_image_enchancement_filter.pdf" document. 
We used python 3 with the numpy and imageio libraries.

The program allows the user to provide parameters in order to read the image and apply the follow methods:
- Bilateral Filter;
- Unsharp mask using the Laplacian Filter, and;
- Vignette Filter.

The input image is grayscale and in the unit8 format.

Input example:

> camera.png<br/>1<br/>0<br/>3<br/>150.0<br/>100.0

Output (Reference/Output Image):

![Reference Image](camera.png)
![Output Image](outputs/case_1_my_output_img.png)
