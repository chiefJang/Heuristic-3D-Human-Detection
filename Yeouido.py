import os
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from mayavi.core import scene
import sys
from scipy.spatial import cKDTree
sys.path.append('../../3DTrans/tools/visual_utils')
import visualize_utils as V
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
import time
# Load 3D point cloud data from a binary file (.bin)

def load_point_cloud_data(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Extract x, y, z coordinates

# Visualize 3D point cloud data
def plot_3d_point_cloud(points, title="3D Point Cloud", c='b'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=c, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


def visualize_mayavi(point_cloud_data,  idx, predicted_boxes=None):
    
    
    new_predicted_boxes = []
    if predicted_boxes is not None:
        for box_info in predicted_boxes:
            center = box_info['center']
            lengths = box_info['lengths']

            # Convert to the new format [x, y, z, dx, dy, dz, heading]
            new_box = [
                center[0],  # x
                center[1],  # y
                center[2],  # z
                lengths[0],  # dx
                lengths[1],  # dy
                lengths[2],  # dz
                0  # heading
            ]

            new_predicted_boxes.append(new_box)
            # Visualize grids using plot3d
     
        final_predicted_boxes = np.array(new_predicted_boxes).reshape(-1, 7)
        V.draw_scenes(
                points=point_cloud_data, ref_boxes=final_predicted_boxes,
                
            )
    else:
        V.draw_scenes(
                    points=point_cloud_data,
                    
                )

    # Show the visualization
    print("===> plt show")
    # scenes = mlab.gcf()
    # scenes_array = mlab.screenshot(figure=scenes)
    # with open('vis.bin', 'wb') as f:
    #     f.write(scenes_array.tobytes())
    #mlab.savefig(filename= './results/' + str(idx) + '_demo.png')
    print('saved Our figure')
    mlab.show()

if __name__ == "__main__":
    
    file_path = 'frm_00000210.bin'
    point_cloud_data = load_point_cloud_data(file_path)
    start = time.time()
    y = point_cloud_data[:, 1].reshape(-1, 1)
    z = point_cloud_data[:, 2].reshape(-1, 1)
    linear = LinearRegression().fit(y, z)
    coef_y, intercept = linear.coef_[0], linear.intercept_
    point_cloud_data[:, 2] = point_cloud_data[:, 2] - (coef_y*point_cloud_data[:, 1] + intercept)
    
    x = point_cloud_data[:, 0].reshape(-1, 1)
    z = point_cloud_data[:, 2].reshape(-1, 1)
    linear = LinearRegression().fit(x, z)
    coef_x, intercept = linear.coef_[0], linear.intercept_
    point_cloud_data[:, 2] = point_cloud_data[:, 2] - (coef_x*point_cloud_data[:, 0] + intercept)

    model = RANSACRegressor()
    model.fit(point_cloud_data[:, :2], point_cloud_data[:, 2])

    # Extract ground points
    ground_mask = model.inlier_mask_
    ground_points = point_cloud_data[ground_mask]
    point_cloud_data = point_cloud_data[~ground_mask]
    ground_z = np.mean(ground_points[:, 2]) + 0.1
    
    #ground_z = -0.9
    max_y = 70
    max_x = 50
    min_x = -50
    max_z = 3.0
    point_cloud_data = point_cloud_data[(point_cloud_data[:, 2] > ground_z-0.5) & (point_cloud_data[:, 2] < max_z)]
    point_cloud_data = point_cloud_data[(point_cloud_data[:, 0] > min_x) & (point_cloud_data[:, 0] < max_x)]
    point_cloud_data = point_cloud_data[(point_cloud_data[:, 1] < max_y)]
    threshhold = 0.1
    mask = (point_cloud_data[:, 2] < ground_z + 0.5 + threshhold) & (point_cloud_data[:, 2] > ground_z + 0.5 - threshhold)
    centerpoints = point_cloud_data[mask]
    eps = 0.2  # 클러스터링 반경 (조절이 필요함)
    min_samples = 1  # 최소 포인트 수

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(centerpoints)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    cluster_centers = np.array([centerpoints[labels == label].mean(axis=0) for label in unique_labels])
    
    
    filtered_label_points_dict = {}
    for idx, center in enumerate(cluster_centers):
        x_center, y_center, z_center = center
        
        # Find points within the specified radius
        radius = 0.25
        mask_radius = (
            (point_cloud_data[:, 0] >= x_center - radius) & (point_cloud_data[:, 0] <= x_center + radius) &
            (point_cloud_data[:, 1] >= y_center - radius) & (point_cloud_data[:, 1] <= y_center + radius)
        )
        
        # Extract points within the radius
        points_within_radius = point_cloud_data[mask_radius]
        
        # Check if there are points with z values greater than 1.5
        mask_z_condition = points_within_radius[:, 2] > ground_z + 2.0
        
        if np.any(mask_z_condition) or points_within_radius.shape[0] < 7:
            # Handle the exception or add to a list of points that meet the condition
            print(f"Exception: Points within radius of {center} have z values greater than 1.5")
            # Your exception handling code here
        else:
            max_z_point_index = np.argmax(points_within_radius[:, 2])
            highest_z_point = points_within_radius[max_z_point_index]
            
            kdtree = cKDTree(point_cloud_data[point_cloud_data[:, 2] > highest_z_point[2]])
            distance, index = kdtree.query(highest_z_point)
            
            if distance > 0.7:
                filtered_label_points_dict[idx] = points_within_radius
    
    
    predicted_boxes = []
    for label, points in filtered_label_points_dict.items():
        # Convert the list of NumPy arrays into a single NumPy array
        points_array = np.stack(points, axis=1)
        
        # Calculate the mean along the specified axis (axis=1 in this case)
        box_center = np.mean(points_array, axis=1)
        
        # Calculate the lengths of the bounding box
        box_lengths = np.max(points_array, axis=1) - np.min(points_array, axis=1)
        if box_lengths[2] < 0.85:
            print("Too small!")
        elif box_lengths[2] > 2.0:
            print("Too tall")
        else:
            print(box_center, box_lengths)
            
            predicted_boxes.append({'center': box_center, 'lengths': box_lengths})
    end = time.time()
    print(end - start)
    visualize_mayavi(point_cloud_data, idx, predicted_boxes)