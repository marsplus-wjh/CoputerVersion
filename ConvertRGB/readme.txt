OS: MacOS 10.14.1

Language: C++

Library: OpenCV 4.0.1

Run: Run in the C++ environment

Features:
	1. Convert the Bayer arrangement(GRAYSCALE) image to RGB components(COLOR) image.
	2. Create the artifacts image to compare original image with demosaicing image.
	3. Improve the algorithm to create the new values of blue layer and green layer 	and compare this new image with previous demosaicing image.

Implementations:
	1. Initial three Mats (Blue, Green, Red), assign the pixels of each point of a 		single channel image to the three Mats in the given order. Then, create kernels
	for each of these three Mats by given convolutions and uses filter2D() to 		calculate them. Finally, merge these three Mat to an image.
	2. Use compute function to visualize the artifacts by given formulas.
	3. Bill Freeman proposed a new algorithm to get Blue and Green value according Red 	value, use medianBlur() to improve results.
