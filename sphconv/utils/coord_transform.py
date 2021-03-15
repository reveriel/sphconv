from typing import List, Tuple
import numpy as np


# transform_func = transform_funcs[coord_system];
# system_points, system_range = transform_func(points, coors_range)

def spherical_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)

    sph_range = np.array([sph_range[0],
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          sph_range[3],
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([r, phi, theta, intensity], 1).astype(points.dtype), sph_range


def lspherical_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)
    sph_range = np.array([np.log(sph_range[0]),
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          np.log(sph_range[3]),
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([np.log(r), phi, theta, intensity], 1).astype(points.dtype), sph_range


def cylinder_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    coor_range = np.array([sph_range[0],
                           np.radians(sph_range[1]),
                           v_range[2],
                           sph_range[3],
                           np.radians(sph_range[4]),
                           v_range[5]])
    return np.stack([d, phi, z, intensity], 1).astype(points.dtype), coor_range


def lcylinder_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    coor_range = np.array([np.log(sph_range[0]),
                           np.radians(sph_range[1]),
                           v_range[2],
                           np.log(sph_range[3]),
                           np.radians(sph_range[4]),
                           v_range[5]])
    return np.stack([np.log(d), phi, z, intensity], 1).astype(points.dtype), coor_range


def hybrid_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)

    sph_range = np.array([sph_range[0],
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          sph_range[3],
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([d, phi, theta, intensity], 1).astype(points.dtype), sph_range


def lhybrid_transform(points: np.array, v_range, sph_range: np.array) -> Tuple[np.array, np.array]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]
    x2y2 = x*x + y*y
    r = np.sqrt(x2y2 + z * z)
    d = np.sqrt(x2y2)
    phi = np.arcsin(y / np.sqrt(x2y2))
    theta = np.arccos(z / r)

    sph_range = np.array([np.log(sph_range[0]),
                          np.radians(sph_range[1]),
                          np.radians(sph_range[2]),
                          np.log(sph_range[3]),
                          np.radians(sph_range[4]),
                          np.radians(sph_range[5])])
    return np.stack([np.log(d), phi, theta, intensity], 1).astype(dtype=points.dtype), sph_range


transform_funcs = {
    "spherical": spherical_transform,
    "l-spherical": lspherical_transform,
    "cylinder": cylinder_transform,
    "l-cylinder": lcylinder_transform,
    "hybrid": hybrid_transform,
    "l-hybrid": lhybrid_transform
}
