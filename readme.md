# Vertical Distortion Correction
This project aims to correct vertical distortion in images.
The code has been tested with Python 3.9.

## Installation
To use this repository, please install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage
To try to correct images you can run in Python the following command:
```bash
python main.py -i input_image -o output_image [--display_lines] [--display_results]
```

The -i and -o arguments are mandatory and represent the input and output image file paths, respectively.
Additionally, there are two optional arguments:
* --display_lines: displays detected lines on the image
* --display_results: displays the corrected image

`Supported extensions for input and output images are jpeg, jpg, png, and tiff.`

## Method
The correction algorithm is based on the detection of vertical vanishing points. Two line detection methods are available: Canny edge detection with Hough transform, or the LineSegmentDetector from OpenCV.

The vanishing point detection and homography estimation is inspired from the paper *Auto-rectification of user photos* by Chaudhury et al. (2014), presented at the IEEE International Conference on Image Processing (ICIP).
