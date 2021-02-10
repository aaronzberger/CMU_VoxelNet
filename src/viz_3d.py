import open3d as o3d


def visualize_lines_3d(gt_boxes, pts):
    '''
    Use Open3D to plot labels on 3D point clouds
    '''

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0] for i in range(len(lines))]

    line_sets = []
    for box in gt_boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)

    pcl = o3d.geometry.PointCloud()

    pcl.points = o3d.utility.Vector3dVector(pts[:, :3].astype('float64'))

    o3d.visualization.draw_geometries([
        pcl, *line_sets
    ])
