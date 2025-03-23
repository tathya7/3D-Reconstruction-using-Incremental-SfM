'''
ENPM 673 - Building built in minutes

Group 21
Tathya Bhatt, Keshav Sharma, Mohammed Munnawar, Kshitij Aggarwal
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from open3d import *    


cwd = os.getcwd()
image_dir = os.path.join(cwd, "monument_dataset")
print(image_dir)

calib_mat = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], 
                      [0 , 2398.118540286656,  628.2649953288065],
                      [0, 0 ,1]]).reshape(3,3)

def get_images(dir):

    img_lst = []
    img_lst = [os.path.join(dir, i)for i in os.listdir(dir)]

    return img_lst

def get_features(frame1, frame2):

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None)

    matches = bf.knnMatch(des1,des2, 2)
    good_features = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_features.append(m)


    # Contains the coordinates of points from the source image which is to be warped and for each match in matches_flann, it retrieves the keypoint coordinates
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_features])
    # Contains the coordinates of points from the Initial image and for each match in matches_flann, it retrieves the keypoint coordinates
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_features])


    return points1, points2

def get_essential_mat(p1,p2, k):
    
    E, em_mask = cv2.findEssentialMat(p1, p2, k, method=cv2.RANSAC, prob=0.99, threshold=0.4, mask=None)

    p1 = p1[em_mask.ravel()==1]
    p2 = p2[em_mask.ravel()==1]

    return E, p1, p2, em_mask

def traingulate_points(proj1, proj2, p1, p2):
    '''
    This function computes the traingulation from 2D Image coordinates to 3D Coordinates
    Further, the 3D coordinates are normalized to convert into homogeneous coordinates
    
    proj1 - Projection Matrix of Camera 1
    proj2 - Projection Matrix of Camera 2
    '''

    world_pts = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    world_pts = world_pts/world_pts[3]
    
    return p1.T, p2.T, world_pts

def find_correspondence(img_points_1, img_points_2, img_points_3):
    '''
    This function finds the correspondence between three consecutive images and stores the 
    corresponding points in an array to further find the points in third image
    '''
    cr_points_1 = []
    cr_points_2 = []

    for i in range(img_points_1.shape[0]):
        a = np.where(img_points_2 == img_points_1[i, :])
        # print("Found correspondences", a)
        if a[0].size != 0:
            cr_points_1.append(i)
            cr_points_2.append(int(a[0][0]))
            # print("MAtch Point Data type", a[0][0].dtype)

    mask_array_1 = np.ma.array(img_points_2, mask=False)
    mask_array_1.mask[cr_points_2] = True
    mask_array_1 = mask_array_1.compressed()
    mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

    mask_array_2 = np.ma.array(img_points_3, mask=False)
    mask_array_2.mask[cr_points_2] = True
    mask_array_2 = mask_array_2.compressed()
    mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)

    return np.array(cr_points_1), np.array(cr_points_2), mask_array_1, mask_array_2

def pnp_ransac(world_pts, p2, k, dist_coeff, rot_vector, first):

    '''
    This function is used to calculate optimal camera poses using multiple views and gives the
    cooresponding rotation and translation matrices

    world_pts - 3D World Points
    p2 - Image Coordinates
    dist_coeff - distortion coefficient if used
    '''
    if first == 1:
        world_pts = world_pts[:, 0 ,:]
        p2 = p2.T
        rot_vector = rot_vector.T 

    
    _, rot_vec, trans_vec, inlier_idx = cv2.solvePnPRansac(world_pts, p2, k, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
    rot_mat, _ = cv2.Rodrigues(rot_vec)

    if inlier_idx is not None:
        p2 = p2[inlier_idx[:, 0]]
        world_pts = world_pts[inlier_idx[:, 0]]
        rot_vector = rot_vector[inlier_idx[:, 0]]

    return rot_mat, trans_vec, p2, world_pts, rot_vec

def reprojection_error(world_pts, proj_mat, k, p2, is_homogeneous):

    rotation_mat = proj_mat[:3,:3]
    rotation_vec, _ = cv2.Rodrigues(rotation_mat)

    if is_homogeneous==1:
        world_pts = cv2.convertPointsFromHomogeneous(world_pts.T)
        
    image_points_obtained, _ = cv2.projectPoints(world_pts, rotation_mat, proj_mat[:3,3], k, None)
    image_points_obtained = np.float32(image_points_obtained[:, 0, :])

    error = cv2.norm(image_points_obtained, np.float32(p2.T) if is_homogeneous == 1 else np.float32(p2), cv2.NORM_L2)

    error = error / len(image_points_obtained)

    return error, world_pts

def to_ply(world_pts, colors):

    '''
    This function is used to convert the point cloud data into ply file
    It was adapted from one of the github repos to visualize the point cloud
    '''
    conv_pts = world_pts.reshape(-1, 3)
    conv_colors = colors.reshape(-1, 3)
    vertical_mat = np.hstack([conv_pts, conv_colors])

    mean = np.mean(vertical_mat[:, :3], axis=0)
    scaled_vertical_mat = vertical_mat[:, :3] - mean
    dist = np.sqrt(scaled_vertical_mat[:, 0] ** 2 + scaled_vertical_mat[:, 1] ** 2 + scaled_vertical_mat[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist))

    vertical_mat = vertical_mat[indx]
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    
    with open('result.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertical_mat)))
        np.savetxt(f, vertical_mat, '%f %f %f %d %d %d')

def view3d_cloud():
    cloud = io.read_point_cloud("result.ply") # Read point cloud
    visualization.draw_geometries([cloud])    # Visualize point cloud      

def view_plot(img_num, error):
    plt.figure(figsize=(8, 5))  # Adjust figure size as needed
    plt.plot(img_num, error, marker='o', linestyle='-')
    plt.xlabel('Image Numbers')
    plt.ylabel('Error')
    plt.title('Reprojection Error with PnP RANSAC')
    plt.grid(True)
    plt.show()

# Getting the images
image_list = get_images(image_dir)

img_num = []
error_pts_after = []

# Creating projection matrix of camera 1 and camera 2, assuming view point 1 as center
proj1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
proj2 = np.empty((3, 4))

# Calculating the respective pose of both cameras
cam_pose1 = np.matmul(calib_mat, proj1)
cam_pose2 = np.empty((3, 4)) 

# Initializing the point cloud array and colors
total_points = np.zeros((1, 3))
total_colors = np.zeros((1, 3))

# Getting thhe first two images to estimate essential matrix
img1 = cv2.imread(image_list[0])
img2 = cv2.imread(image_list[1])

# Finding features between images
feature_pts1, feature_pts2 = get_features(img1, img2)

# Getting essential matrix and inlier mask
ematrix, feature_pts1, feature_pts2,  emask = get_essential_mat(feature_pts1, feature_pts2, calib_mat)
# # print("Essential Matrix", essential_matrix)

# Decomposing the essential matrix into rotation and translation
_, init_rot_mat, init_trans_mat, em_mask = cv2.recoverPose(ematrix, feature_pts1, feature_pts2, calib_mat)

# Calculating the projection matrix of view 2
proj2[:3, :3] = np.matmul(init_rot_mat, proj1[:3, :3])
proj2[:3, 3] = proj1[:3, 3] + np.matmul(proj1[:3, :3], init_trans_mat.ravel())

cam_poses = calib_mat.ravel()
# Estimating camera poseses
cam_pose1 = np.matmul(calib_mat, proj2)

# Performing traingulation
feature_pts1, feature_pts2, world_pts = traingulate_points(cam_pose1, cam_pose2, feature_pts1, feature_pts2)
# print("Features points 2 after traingulation",len(feature_1 ))

# Calculating reprojection error
error, world_pts = reprojection_error(world_pts, proj2, calib_mat, feature_pts2, is_homogeneous = 1)

_, _, feature_pts2, world_pts, _ = pnp_ransac(world_pts, feature_pts2, calib_mat, np.zeros((5, 1), dtype=np.float32), feature_pts1, first=1)


total_images = len(image_list) - 2 
cam_poses = np.hstack((np.hstack((cam_poses, cam_pose1.ravel())), cam_pose2.ravel()))

for i in range(len(image_list)-2):

    img3 = cv2.imread(image_list[i + 2])
    print("Current Image Number", i+2)

    features_cur, features_next = get_features(img2, img3)

    if i != 0:

        # Need to avoid first two images as we already calculated their world points
        feature_pts1, feature_pts2, world_pts = traingulate_points(cam_pose1, cam_pose2, feature_pts1, feature_pts2)
        feature_pts2 = feature_pts2.T
        world_pts = cv2.convertPointsFromHomogeneous(world_pts.T)
        world_pts = world_pts[:, 0, :]

    corres_img1_idx, corres_img2_idx, corres_pts1, corres_pts2 = find_correspondence(feature_pts2, features_cur, features_next)
    # print("Corresponding Points Frame2", corres_img2_idx)
    corres_pts_next = features_next[corres_img2_idx]
    # print("Corresponding Points Frame2", corres_img2_idx)
    corres_pts_cur = features_cur[corres_img2_idx]


    rot_matrix, tran_matrix, corres_pts_next, world_pts, corres_pts_cur = pnp_ransac(world_pts[corres_img1_idx], corres_pts_next, calib_mat, np.zeros((5, 1), dtype=np.float32),corres_pts_next, first= 0)

    proj_mat_next = np.hstack((rot_matrix, tran_matrix))
    cam_pos_next = np.matmul(calib_mat, proj_mat_next)


    error, world_pts = reprojection_error(world_pts, proj_mat_next,calib_mat,  corres_pts_next, is_homogeneous = 0)

    corres_pts1, corres_pts2, world_pts = traingulate_points(cam_pose2, cam_pos_next, corres_pts1, corres_pts2)
    error, world_pts = reprojection_error(world_pts, proj_mat_next, calib_mat, corres_pts2, is_homogeneous = 1)
    print("Reprojection Error: ", error)

    error_pts_after.append(error)
    img_num.append(i)

    cam_poses = np.hstack((cam_poses, cam_pos_next.ravel()))



    total_points = np.vstack((total_points, world_pts[:, 0, :]))
    points_left = np.array(corres_pts2, dtype=np.int32)
    color_vector = np.array([img3[l[1], l[0]] for l in points_left.T])
    total_colors = np.vstack((total_colors, color_vector)) 
    proj1 = np.copy(proj_mat_next)
    cam_pose1 = np.copy(cam_pose2)
    img1 = np.copy(img2)
    img2 = np.copy(img3)
    feature_pts1 = np.copy(features_cur)
    feature_pts2 = np.copy(features_next)
    cam_pose2 = np.copy(cam_pos_next)

to_ply(total_points, total_colors)
view3d_cloud()
view_plot(img_num, error_pts_after)