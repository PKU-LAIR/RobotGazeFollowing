import open3d as o3d
import numpy as np
from math import radians, sin, cos


def generate_color_lst(num_of_class):
    color_lst = list()
    for i in range(num_of_class):
        color = np.random.choice(range(256), size=3) / 256
        color_lst.append(color)
    return color_lst


class Visualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.vis.add_geometry(coordinate_frame)
        self.vis.update_geometry(coordinate_frame)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.pcd = o3d.geometry.PointCloud()
        self.pointcloud_colors = None

    def add_attention(self, start_point, yaw, pitch):
        arrow = o3d.geometry.TriangleMesh.create_arrow(0.02, 0.04, 2.0, 0.5)
        R = np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        arrow.rotate(R, [0, 0, 0])
        R = np.asarray([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
        arrow.rotate(R, [0, 0, 0])
        pitch = -pitch
        R = np.asarray([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
        arrow.rotate(R, [0, 0, 0])
        arrow.translate(start_point)
        arrow.paint_uniform_color([0, 0, 0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.07)
        sphere.translate(start_point)
        sphere.paint_uniform_color([0, 0, 0])

        self.vis.add_geometry(arrow)
        self.vis.update_geometry(arrow)
        self.vis.add_geometry(sphere)
        self.vis.update_geometry(sphere)
        self.vis.poll_events()
        self.vis.update_renderer()

    def add_points(self, points, ids):
        self.pcd.points = o3d.utility.Vector3dVector(points)
        color_lst = generate_color_lst(100)
        self.pointcloud_colors = np.asarray([color_lst[idx] for idx in ids])
        self.pcd.colors = o3d.utility.Vector3dVector(self.pointcloud_colors)

        self.vis.add_geometry(self.pcd)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_points_colors(self, update_idx):
        self.pointcloud_colors = np.zeros(np.shape(self.pointcloud_colors))
        self.pointcloud_colors[update_idx, :] = np.asarray([1, 0, 0])
        self.pcd.colors = o3d.utility.Vector3dVector(self.pointcloud_colors)

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_renderer(self):
        self.vis.poll_events()
        self.vis.update_renderer()
