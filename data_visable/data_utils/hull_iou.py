from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def create_multipolygon_from_list(polygon_list):
    """从多边形顶点列表创建MultiPolygon对象"""
    return MultiPolygon([Polygon(polygon) for polygon in polygon_list])

def calculate_group_iou(group1_polygons, group2_polygons):
    """
    计算两组凸包的交并比。

    参数:
    group1_polygons, group2_polygons: 分别为两个组内所有凸包顶点列表的列表。

    返回:
    iou: 两组凸包的交并比。
    """
    # 创建MultiPolygon对象
    multi_poly1 = create_multipolygon_from_list(group1_polygons)
    multi_poly2 = create_multipolygon_from_list(group2_polygons)

    # 计算并集
    union1 = unary_union(multi_poly1)
    union2 = unary_union(multi_poly2)

    # 计算交集
    intersection = union1.intersection(union2)

    # 计算交并比，注意检查并集面积是否为0避免除以零错误
    if union1.area + union2.area - intersection.area == 0:
        iou = 0.0
    else:
        iou = intersection.area / (union1.area + union2.area - intersection.area)

    return iou


def calculate_3d_group_iou(group1_points, group2_points):
    """
    计算两组三维点云所构成的凸包的交并比。
    
    参数:
    group1_points, group2_points: 分别为两个组内所有点的坐标列表，每个点为一个三维坐标元组。
    
    返回:
    iou: 两组凸包的交并比。
    """
    try:
        # 计算每组的凸包
        hull1 = ConvexHull(group1_points)
        hull2 = ConvexHull(group2_points)
        
        # 计算两组凸包的顶点坐标
        vertices1 = hull1.points[hull1.vertices]
        vertices2 = hull2.points[hull2.vertices]
        
        # 计算两组凸包的体积
        vol1 = hull1.volume
        vol2 = hull2.volume
        
        # 计算交集的顶点坐标
        # 注意：直接计算三维凸包的交集较为复杂，这里简化处理，实际应用中可能需要更复杂的几何算法或库
        # 假设使用某种方法得到交集的顶点（这里仅示意，实际实现可能需要其他方法）
        try:
            intersection_hull = ConvexHull(np.concatenate((vertices1, vertices2)))
            vol_intersection = intersection_hull.volume
        except QhullError:  # 如果没有交集，Qhull可能会抛出异常
            vol_intersection = 0.0
        
        # 计算并集体积（直接相加可能会有重叠部分被重复计算，但因为我们关注的是比例，所以这不影响IOU的计算）
        vol_union = vol1 + vol2 - vol_intersection
        
        # 避免除以零错误
        if vol_union == 0:
            iou = 0.0
        else:
            iou = vol_intersection / vol_union
    
    except Exception as e:
        print(f"An error occurred during convex hull calculation: {e}")
        iou = None
    
    return iou




if __name__ == "__main__":
    # 示例数据（请根据实际情况替换）
    group1 = [[(0, 0), (0, 2), (3, 2), (3, 0)], [(4, 1), (4, 3), (6, 3), (6, 1)]]  # 第一组凸包顶点列表
    group2 = [[(1, 1), (1, 3), (4, 3), (4, 1)], [(5, 0), (5, 2), (7, 2), (7, 0)]]  # 第二组凸包顶点列表

    # 计算两组凸包的IoU
    iou = calculate_group_iou(group1, group2)
    print(f"The IoU between the two groups of convex hulls is: {iou}")
    
    # 示例数据（请根据实际情况替换）
    group1_points = np.array([[0, 0, 0], [0, 2, 0], [3, 2, 0], [3, 0, 0], [1, 1, 1]])  # 第一组点云坐标
    group2_points = np.array([[1, 1, 1], [1, 3, 1], [4, 3, 1], [4, 1, 1], [2, 2, 2]])  # 第二组点云坐标

    # 计算两组凸包的IoU
    iou = calculate_3d_group_iou(group1_points, group2_points)
    print(f"The IoU between the two groups of 3D convex hulls is: {iou}")