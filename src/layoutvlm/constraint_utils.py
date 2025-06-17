import torch
import numpy as np
from shapely.geometry import Polygon, Point
from utils.placement_utils import half_vector_intersects_polygon
import torch.nn.functional as F

    
def point_to_segment_batch_loss(points, segments):
    """
    Calculate the shortest distance between a batch of points and a batch of line segments in a differentiable manner using PyTorch.

    Parameters:
    points (Tensor): A batch of 2D points of shape [N, 2].
    segments (Tensor): A batch of line segments of shape [M, 4].

    Returns:
    Tensor: A tensor of shape [N, M] containing the shortest distances from each point to each line segment.
    """
    px, py = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1)
    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]

    # Reshape for broadcasting
    x1, y1, x2, y2 = x1.unsqueeze(0), y1.unsqueeze(0), x2.unsqueeze(0), y2.unsqueeze(0)

    # Vector from the first endpoint to the points
    dpx = px - x1
    dpy = py - y1

    # Vector from the first endpoint to the second endpoint
    dx = x2 - x1
    dy = y2 - y1

    # Dot product of the above vectors
    dot_product = dpx * dx + dpy * dy

    # Length squared of the segment vector
    len_sq = dx * dx + dy * dy

    # Projection factor normalized to [0, 1]
    projection = dot_product / (len_sq + 1e-8)
    projection = torch.clamp(projection, 0, 1)

    # Closest points on the segments
    closest_x = x1 + projection * dx
    closest_y = y1 + projection * dy

    #closest = torch.concat([closest_x, closest_y], dim=-1)
    #pp = torch.concat([px, py], dim=-1)

    # Distance from the points to the closest points on the segments
    #distance = torch.sqrt((closest_x - px) ** 2 + (closest_y - py) ** 2)
    distance = (closest_x - px) ** 2 + (closest_y - py) ** 2
    return distance

def periodic_sin_loss(vector1, vector2):
    """
    Calculate the custom loss based on the cosine similarity, which is zero
    at 90 and  270 degrees and increases otherwise.

    Args:
        vector1 (torch.Tensor): First vector tensor.
        vector2 (torch.Tensor): Second vector tensor.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Normalize the vectors
    vector1_norm = F.normalize(vector1, p=2, dim=-1)
    vector2_norm = F.normalize(vector2, p=2, dim=-1)
    
    # Calculate the cosine similarity
    cosine_similarity = torch.sum(vector1_norm * vector2_norm, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    
    # Convert cosine similarity to radians
    angles = torch.acos(cosine_similarity)
    
    # Compute the periodic loss
    #loss = torch.cos(angles)**2
    loss = abs(torch.sin(angles))
    
    return loss.mean()

def periodic_loss(vector1, vector2, epsilon=1e-7):
    """
    Calculate the custom loss based on the cosine similarity, which is zero
    at 0, 90, 180, and 270 degrees and increases otherwise.

    Args:
        vector1 (torch.Tensor): First vector tensor.
        vector2 (torch.Tensor): Second vector tensor.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Normalize the vectors
    vector1_norm = F.normalize(vector1, p=2, dim=-1)
    vector2_norm = F.normalize(vector2, p=2, dim=-1)
    # Calculate cosine similarity, add clamping for numerical stability
    cosine_similarity = torch.sum(vector1_norm * vector2_norm, dim=-1).clamp(-1 + epsilon, 1 - epsilon)
    # Compute the angle in radians
    cos_2theta = 2 * cosine_similarity**2 - 1
    # Use cos(2 * angle) to create periodic loss zero at 0, 90, 180, and 270 degrees
    loss = 1 - cos_2theta**2
    # Return the mean loss
    return loss.mean()

def cosine_distance_loss(vector1, vector2, epsilon=1e-5, beta=10):
    """
    Calculate the loss based on the cosine distance between two vectors.

    Args:
        vector1 (torch.Tensor): First vector tensor.
        vector2 (torch.Tensor): Second vector tensor.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Normalize the vectors
    vector1_norm = F.normalize(vector1, p=2, dim=-1)
    # apply small perturbation?
    # vector1_norm = vector1_norm + epsilon * torch.sign(vector1_norm)
    vector2_norm = F.normalize(vector2, p=2, dim=-1)

    # Calculate cosine similarity
    cosine_similarity = torch.sum(vector1_norm * vector2_norm, dim=-1)
    return 1 - cosine_similarity.mean()
    #cosine_similarity = cosine_similarity.clamp(-1 + epsilon, 1 - epsilon)
    #return F.softplus(-beta * cosine_similarity).mean()
    # Convert cosine similarity to cosine distance (1 - cosine similarity)
    #return (-cosine_similarity).mean()

def angle_between(v1, v2):
    unit_v1 = v1 / torch.norm(v1)
    unit_v2 = v2 / torch.norm(v2)
    dot_product = torch.dot(unit_v1, unit_v2)
    return torch.acos(dot_product)


# def distance_loss(coord1, coord2, min_distance=1.0, max_distance=3.0):
#     """
#     Calculate the loss based on the distance between two coordinates being within a specific range.

#     Args:
#         coord1 (torch.Tensor): First coordinate tensor.
#         coord2 (torch.Tensor): Second coordinate tensor.
#         min_distance (float): The minimum distance threshold. Default is 1.0.
#         max_distance (float): The maximum distance threshold. Default is 3.0.

#     Returns:
#         torch.Tensor: The computed loss.
#     """
#     # Calculate the squared Euclidean distance between the two coordinates
#     distance = torch.sqrt(torch.sum((coord1 - coord2) ** 2))

#     # Use differentiable operations to calculate loss based on the distance range
#     below_min_loss = F.relu(min_distance - distance) ** 2
#     above_max_loss = F.relu(distance - max_distance) ** 2
#     within_range_loss = torch.tensor(0.00, dtype=distance.dtype, device=distance.device)

#     # Select the appropriate loss
#     loss = torch.where(
#         distance < min_distance,
#         below_min_loss,
#         torch.where(distance > max_distance, above_max_loss, within_range_loss)
#     )

#     return loss

def distance_loss(coord1, coord2, min_distance=1.0, max_distance=3.0):
    """
    Calculate the loss based on the distance between two coordinates being within a specific range.

    Args:
        coord1 (torch.Tensor): First coordinate tensor.
        coord2 (torch.Tensor): Second coordinate tensor.
        min_distance (float): The minimum distance threshold. Default is 1.0.
        max_distance (float): The maximum distance threshold. Default is 3.0.

    Returns:
        torch.Tensor: The computed loss.
    """
    min_distance = 0 if min_distance is None else min_distance
    max_distance = 1e6 if max_distance is None else max_distance

    # Calculate the squared Euclidean distance between the two coordinates
    squared_distance = torch.sum((coord1 - coord2) ** 2)
    
    # Compute the loss based on the distance range using smooth approximations
    below_min_loss = F.relu(min_distance**2 - squared_distance)
    above_max_loss = F.relu(squared_distance - max_distance**2)
    # if 
    
    loss = below_min_loss + above_max_loss
    return loss


def does_intersect(asset1, asset2):
    poly1 = asset1.get_polygon()
    poly2 = asset2.get_polygon()
    
    def get_edges(polygon):
        return [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
    
    def project(polygon, axis):
        dots = torch.matmul(polygon, axis)
        return torch.min(dots), torch.max(dots)

    def sigmoid_approx(x, k=10):
        return torch.sigmoid(k * x)
    
    def overlap(min1, max1, min2, max2):
        # Use sigmoid to approximate the step function for differentiability
        return sigmoid_approx(max2 - min1) * sigmoid_approx(max1 - min2)
    
    def get_normals(edges):
        return [torch.tensor([-(edge[1][1] - edge[0][1]), edge[1][0] - edge[0][0]]) for edge in edges]
    
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)
    
    axes = get_normals(edges1) + get_normals(edges2)
    
    #return torch.sum(poly1) + (asset1.position[0] <= 1)
    #return torch.sum(axes)
    intersection_score = 1.0  # Start with full overlap (score of 1)
    for axis in axes:
        min1, max1 = project(poly1, axis)
        min2, max2 = project(poly2, axis)
        #import pdb;pdb.set_trace()
        intersection_score *= overlap(min1, max1, min2, max2)

    # Loss should be 0 if there is no intersection and positive otherwise
    #return 1 - intersection_score  # Loss should be 0 if there is no intersection and positive otherwise
    #return intersection_score  # Loss should be 0 if there is no intersection and positive otherwise

    #target_distance = 6.
    #distance = torch.sqrt(torch.sum((asset1.position - asset2.position) ** 2))
    # Calculate the mean squared error between the distance and the target distance
    #loss = F.mse_loss(distance, torch.tensor(target_distance))
    return F.mse_loss(asset1.rotation, asset2.rotation + np.pi/2)


def is_point_on_line_segment(point, line_start, line_end):
    """Check if a point lies on a line segment."""
    return (np.cross(line_end - line_start, point - line_start) == 0 and 
            np.dot(line_end - line_start, point - line_start) >= 0 and 
            np.dot(line_start - line_end, point - line_end) >= 0)

def ray_intersects_segment(origin, direction, v1, v2):
    """Check if a ray intersects with a line segment."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Check if the ray's origin is on the line segment
    if is_point_on_line_segment(origin, v1, v2):
        return True
    
    # Calculate the intersection point
    v = v2 - v1
    cross_product = np.cross(direction, v)
    
    # Check if the ray is parallel to the line segment
    if abs(cross_product) < 1e-8:
        return False
    
    t = np.cross(v1 - origin, v) / cross_product
    u = np.cross(direction, origin - v1) / cross_product
    
    # Check if the intersection point is on the line segment and in the direction of the ray
    return t >= 0 and 0 <= u <= 1

def ray_intersects_polygon(origin, direction, polygon):
    """Check if a ray intersects with a polygon."""
    origin = np.array(origin)
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)  # Normalize direction
    
    for i in range(len(polygon)):
        if ray_intersects_segment(origin, direction, polygon[i], polygon[(i + 1) % len(polygon)]):
            return True
    
    return False