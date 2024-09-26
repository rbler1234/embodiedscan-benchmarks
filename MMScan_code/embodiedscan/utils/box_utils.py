import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as array of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, np.split(euler_angles, 3, axis=-1))
    ]
    matrices = [x.squeeze(axis=-3) for x in matrices]
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])

def is_inside_box(points, center, size, rotation_mat):
    """
        Check if points are inside a 3D bounding box.
        Args:
            points: 3D points, numpy array of shape (n, 3).
            center: center of the box, numpy array of shape (3, ).
            size: size of the box, numpy array of shape (3, ).
            rotation_mat: rotation matrix of the box, numpy array of shape (3, 3).
        Returns:
            Boolean array of shape (n, ) indicating if each point is inside the box.
    """
    assert points.shape[1] == 3, "points should be of shape (n, 3)"
    center = np.array(center) # n, 3
    size = np.array(size) # n, 3
    rotation_mat = np.array(rotation_mat)
    assert rotation_mat.shape == (3, 3), f"R should be shape (3,3), but got {rotation_mat.shape}"
    # pcd_local = (rotation_mat.T @ (points - center).T).T  The expressions are equivalent
    pcd_local = (points - center) @ rotation_mat # n, 3
    pcd_local = pcd_local / size * 2.0  # scale to [-1, 1] # n, 3
    pcd_local = abs(pcd_local)
    return (pcd_local[:, 0] <= 1) & (pcd_local[:, 1] <= 1) & (pcd_local[:, 2] <= 1)

def normalize_box(scene_pcd, embodied_scan_bbox):
    bbox = np.array(embodied_scan_bbox)
    orientation=R.from_euler("zxy",bbox[6:],degrees = False).as_matrix().tolist()
    position=np.array(bbox[:3])
    size=np.array(bbox[3:6])
    obj_mask = torch.tensor(is_inside_box(scene_pcd[:,:3],position,size,orientation),dtype=bool)
    obj_pc = scene_pcd[obj_mask]
    
    
    # resume the same if there's None
    if obj_pc.shape[0]<1:
        return embodied_scan_bbox
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
    return bbox

def __9DOF_to_6DOF__(pcd_data, bbox_):
    
    #that's a kind of loss of information, so we don't recommend
    return normalize_box(pcd_data[0], bbox_)