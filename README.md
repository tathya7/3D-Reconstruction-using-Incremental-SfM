# 3D Reconstruction from Multiple Images
![alt text](https://github.com/tathya7/3D-Reconstruction-using-Incremental-SfM/blob/main/image.png)
## Overview

This Python script performs 3D reconstruction of a scene from a set of images using Structure from Motion (SfM) techniques.
The script utilizes OpenCV for feature extraction, essential matrix estimation, and camera pose estimation.  Additionally, it uses Open3D for visualizing the resulting point cloud.

## Dependencies

*   opencv-python
*   numpy
*   matplotlib
*   open3d

You can install these dependencies using pip: `pip install opencv-python numpy matplotlib open3d`


## Usage

1.  **Prepare your image dataset:** Create a directory named `your_dataset` in the same directory as the script. Place the images you want to use for reconstruction in this directory.
2.  **Run the script:** `python main.py`



