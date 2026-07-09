"""
This module provides functionality for generating neuromaps from 3D voxel data, including
projection, morphing, saliency map generation, and contour extraction. It defines classes and
methods to handle the processing of voxel positions, loadings, and atlas labels to produce
visual representations of brain activity and structure.
"""
from __future__ import annotations
from enum import StrEnum, IntEnum
import numpy.typing as npt
import numpy as np
from scipy.signal import savgol_filter
import cv2
import cc3d


class Projection:
    """
    Class for projecting 3D voxel positions onto a 2D plane based on specified parameters.
    """

    class ConnectivityType(StrEnum):
        """
        Enum for specifying the type of connectivity used for determining connected components in 
        the projection process.
        """
        LABEL = "label"
        CONNECTIVITY6 = "connectivity6"
        CONNECTIVITY18 = "connectivity18"
        CONNECTIVITY26 = "connectivity26"

    DEFAULT_ACTIVITY_THRESHOLD: float = 10
    DEFAULT_CLUSTER_SIZE_THRESHOLD: int = 200
    DEFAULT_CONNECTIVITY: ConnectivityType = ConnectivityType.LABEL
    DEFAULT_SORT_BY_EIGENVALUE: bool = False

    # does not consider weighting based on loading -> very low loading equally threated as high one
    @staticmethod
    def project(
        atlas: list[int],
        loading: list[float],
        activity_threshold: float,
        locations: npt.NDArray[np.float32],
        shape: tuple[int],
        cluster_size_threshold: int,
        connectivity: Projection.ConnectivityType,
        sort_by_eigenvalue: bool) -> npt.NDArray[np.float32]:
        """
        Computes the projection vector to project the 3D voxel positions onto a 2D plane based on 
        the provided atlas, loading values, and specified parameters.
        
        Args:
            atlas: A list of integers representing the atlas labels for each voxel.
            loading: A list of floats representing the loading values for each voxel.
            activity_threshold: A float threshold for activity; values below this are ignored.
            locations: A 2D array of shape (N, 3) representing the (x, y, z) coordinates of the 
                voxels.
            shape: A tuple representing the dimensions of the 3D volume.
            cluster_size_threshold: An integer threshold for cluster size in projection.
            connectivity: A string indicating the type of connectivity for projection.
            sort_by_eigenvalue: A boolean indicating whether to sort projection clusters 
                by eigenvalue.

        Raises:
            RuntimeError: If the activity_threshold results in an empty loading mask.

        Returns:
            A 2D array representing the projection vector.
        """
        mask: npt.NDArray[np.bool_] = np.abs(loading)>activity_threshold
        if not np.sum(mask):
            raise RuntimeError("Too high activity_threshold results in empty loading mask")

        clusters: dict[int, npt.NDArray[np.float32]] = Projection._get_projection_clusters(
            atlas, mask, locations, shape, cluster_size_threshold, connectivity
        )
        return Projection._calc_proj_vec(clusters, sort_by_eigenvalue)

    @staticmethod
    def _calc_proj_vec(
        x_n: dict[int, npt.NDArray[np.float32]],
        sort_by_eigenvalue: bool) -> npt.NDArray[np.float32]:
        """
        Computes the projection vector based on the provided clusters of voxel positions.
        
        Args:
            x_n: A dictionary mapping each label to its corresponding array of (x, y, z) 
                coordinates.
            sort_by_eigenvalue: A boolean indicating whether to sort projection dimensions by 
                eigenvalue.
        
        Returns:
            A 2D array representing the projection vector.
        """
        x_stacked = np.vstack(list(x_n.values()))

        priori_probs = [len(x)/len(x_stacked) for x in x_n.values()]
        s_w = np.zeros((3,3))
        mu_l = {l:1/len(x)*np.sum(x, axis=0) for l,x in x_n.items()}
        for l,x in x_n.items():
            s_w += (x - np.array(len(x)*[mu_l[l]])).T@(x - np.array(len(x)*[mu_l[l]]))

        # between-class scatter matrix
        s_b = np.zeros((3,3))
        mus= list(mu_l.values())
        for i in range(len(x_n)-1):
            for j in range(i+1, len(x_n)):
                p_i = priori_probs[i]
                p_j = priori_probs[j]
                m_i = mus[i]
                m_j = mus[j]
                s_b += p_i*p_j*np.outer((m_i-m_j).T, (m_i-m_j))

        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(s_w)*s_b)
        if sort_by_eigenvalue:
            eigvecs = eigvecs[np.argsort(np.real(eigvals))[::-1],:]
        return eigvecs[:2]

    @staticmethod
    def _get_projection_clusters(
        atlas: npt.NDArray[np.int32],
        mask: npt.NDArray[np.bool_],
        positions: npt.NDArray[np.float32],
        shape: tuple[int],
        cluster_size_threshold: int,
        connectivity: Projection.ConnectivityType
        ) -> dict[int, npt.NDArray[np.float32]]:

        loading_mask = (atlas>0) & mask
        components = []
        match(connectivity):
            case Projection.ConnectivityType.LABEL:
                components = {
                    l: np.array(np.where(((atlas==l) & loading_mask).reshape(shape))).T
                    for l in np.unique(atlas) if l > 0
                }
            case Projection.ConnectivityType.CONNECTIVITY6 | \
                 Projection.ConnectivityType.CONNECTIVITY18 | \
                 Projection.ConnectivityType.CONNECTIVITY26:
                mask = atlas.copy().astype(bool)
                mask[~loading_mask] = 0
                neighborhood_size: int = 6
                if connectivity==Projection.ConnectivityType.CONNECTIVITY18:
                    neighborhood_size = 18
                elif connectivity==Projection.ConnectivityType.CONNECTIVITY26:
                    neighborhood_size = 26
                labels: npt.NDArray[np.int32] = cc3d.connected_components(
                    mask.reshape(shape),
                    connectivity=neighborhood_size
                )
                components = {
                    l: positions[(labels.flatten()==l) & loading_mask]
                    for l in np.unique(atlas) if l > 0
                }
            case _:
                raise RuntimeError("Unsupported connectivity type")
        return {l: pos for l,pos in components.items() if len(pos)>cluster_size_threshold}

class Morphing:
    """
    Class for morphing the voxel positions of a 3D volume based on specified parameters.
    """

    DEFAULT_BETA: float = .2
    DEFAULT_Z: float = 2
    DEFAULT_SIGMA: float = .2

    _VOLUME_ORIGIN: npt.NDArray[np.int32] = np.array([0, 0, -1])

    class Coords(IntEnum):
        """
        Enum for indexing the coordinates in a 3D space.
        """
        X = 0
        Y = 1
        Z = 2

    @staticmethod
    def _get_normalized_positions(shape: tuple[int]):
        idx = np.indices(shape)
        idx = idx.reshape(idx.shape[Morphing.Coords.X], -1).T
        def map_pts(x):
            pts = 2*(1.0/shape[0]*np.array(x) - 0.5)
            pts[Morphing.Coords.Z] *= -1
            pts[Morphing.Coords.Z] += -1 - np.min(pts[2])
            return pts
        return map_pts(idx.T).T

    @staticmethod
    def get_morph(shape: tuple[int],
                  beta: float,
                  z_scale: float,
                  sigma: float) -> npt.NDArray[np.float32]:
        """
        Morphs the voxel positions of a 3D volume based on the provided parameters.

        Args:
            shape: A tuple representing the dimensions of the 3D volume.
            beta: A float parameter controlling the degree of morphing.
            z_scale: A float parameter controlling the scaling in the z-direction.
            sigma: A float parameter controlling the radial expansion of the morph.
        
        Returns:
            A 2D array of shape (N, 3) containing the morphed positions of the voxels, where N is 
                the total number of voxels in the volume.
        """
        p = Morphing._get_normalized_positions(shape)
        theta = np.arctan2((p[:,Morphing.Coords.X]**2+p[:,Morphing.Coords.Y]**2)**(1/2),
                           p[:,Morphing.Coords.Z])

        p_proj = p-Morphing._VOLUME_ORIGIN
        psi_0 = np.arccos(
            (p[:,Morphing.Coords.X]**2+p[:,Morphing.Coords.Y]**2)**(1/2)
            / (p_proj[:,Morphing.Coords.X]**2+p_proj[:,Morphing.Coords.Y]**2
               +p_proj[:,Morphing.Coords.Z]**2)**(1/2))
        psi_beta = -beta*psi_0 + psi_0
        r_0 = np.linalg.norm(p_proj, axis=1)
        r_e = np.pi-theta
        r_beta = beta*(r_e-r_0) + r_0

        p_z = np.tan(psi_beta) * np.linalg.norm(p[:,:Morphing.Coords.Z], axis=1)
        p_t_dir = np.hstack((p[:,:2], p_z[:,None]))
        p_t_norm = p_t_dir/np.linalg.norm(p_t_dir, axis=1)[:,None]
        p_t = Morphing._VOLUME_ORIGIN + r_beta[:,None]*p_t_norm

        # eviscerate movement in z-direction
        evisc_w_z = z_scale/2 * (1+p[:,Morphing.Coords.Z])
        evisc_w_xy = np.exp(-np.linalg.norm(p[:,:2], axis=1)**2/(2*sigma))
        p_z_max_evisc = evisc_w_xy * evisc_w_z
        p_t[:,Morphing.Coords.Z] += -beta*p_z_max_evisc
        return p_t

class Saliency:
    """
    Class for generating saliency maps from 2D point locations and their corresponding loadings.
    """

    DEFAULT_GRID_SIZE: int = 40

    @staticmethod
    def get_saliency_map(locations: npt.NDArray[np.float32],
                         weights: npt.NDArray[np.float32],
                         grid_size: int) -> npt.NDArray[np.float32]:
        """
        Generate a saliency map based on the provided 2D point locations and their corresponding 
        weights.
        Fails if the locations are not normalized to [0, 1].

        Args:
            locations: A 2D array of shape (N, 2) representing the (x, y) coordinates of the points.
            weights: A 1D array of shape (N,) representing the weights (loadings) associated with 
                each point.
            grid_size: An integer representing the size of the grid for the quadratic saliency map.
        
        Returns:
            A 2D array of shape (2, grid_size, grid_size) containing the saliency map and the 
                corresponding alpha values, where the first channel represents the saliency values 
                and the second channel represents the alpha values.
        """
        assert np.all(np.min(locations, axis=0)+np.finfo(np.float32).eps >= 0)
        assert np.all(np.max(locations, axis=0)-np.finfo(np.float32).eps <= 1)

        total_salience, _, _ = np.histogram2d(
            locations[:,1], locations[:,0],
            bins=[grid_size, grid_size],
            range=[[0, 1], [0, 1]],
            weights=np.abs(weights)
        )
        alpha = (total_salience/np.max(total_salience))**(.8)
        salience, _, _ = np.histogram2d(
            locations[:,1], locations[:,0],
            bins=[grid_size, grid_size],
            range=[[0, 1], [0, 1]],
            weights=weights
        )
        # clip values
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        return np.array([salience.astype(np.float32), alpha.astype(np.float32)])

class Contour:
    """
    Class for generating contours from 2D point locations.
    """
    DEFAULT_DETECTION_SCALE: int = 80
    DEFAULT_SPACING: int = 2
    DEFAULT_SMOOTHING_WINDOW: int = 11
    DEFAULT_SMOOTHING_POLYNOM: int = 3
    DEFAULT_CUTOFF_THRESH: int = 20
    DEFAULT_CLOSING_KERNEL: tuple[int,int]=(3,3)

    @staticmethod
    def get_contours(locations: dict[int, npt.NDArray[np.float32]],
                     detection_scale: int,
                     spacing: int,
                     smoothing_window: int,
                     smoothing_polynom: int,
                     cutoff_thresh: int,
                     closing_kernel: tuple[int,int]) -> dict[int,list[npt.NDArray[np.float32]]]:
        """
        Generate contours for each label based on the provided locations.
        Fails if the locations are not normalized to [0, 1].

        Args:
            locations: A dictionary mapping each label to its corresponding array of (x, y)
                coordinates.
            detection_scale: An integer scale for contour detection.
            spacing: An integer spacing for contour resampling.
            smoothing_window: An integer window size for contour smoothing.
            smoothing_polynom: An integer polynomial order for contour smoothing.
            cutoff_thresh: An integer threshold for contour cutoff; contours with their number of 
                points below this threshold are ignored.
            closing_kernel: A tuple representing the kernel size for contour closing.

        Returns:
            A dictionary mapping each label to its corresponding list of contour arrays.
        """

        for locs in locations.values():
            assert np.all(np.min(locs, axis=0)+np.finfo(np.float32).eps >= 0)
            assert np.all(np.max(locs, axis=0)-np.finfo(np.float32).eps <= 1)

        def contours_from_positions(pos: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
            label_idxs = np.ravel_multi_index((np.round(pos*(detection_scale-1))).astype(int).T,
                                              (detection_scale, detection_scale))
            mask = np.zeros(detection_scale**2, dtype=np.uint8)
            mask[label_idxs] = 1
            mask = mask.reshape(detection_scale, detection_scale)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
            contours, _ = cv2.findContours(
                mask.T,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE
            )

            pruned_contours = []
            for contour in contours:
                contour = contour.reshape(-1,2)
                resampled = Contour._resample_by_distance(contour, spacing)
                if len(resampled) > cutoff_thresh:
                    pruned_contours += [
                        Contour._smooth_contour(resampled, smoothing_window, smoothing_polynom)
                    ]
            return pruned_contours

        labels_contours: dict[int,list[npt.NDArray[np.float32]]] = {}
        for l,xy in locations.items():
            labels_contours[l] = [c/detection_scale for c in contours_from_positions(xy)]
        return labels_contours

    @staticmethod
    def norm_space(x: npt.NDArray[np.float32],
                   mins: npt.NDArray[np.float32],
                   maxs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        MinMax normalization of the input array `x` to the range [0, 1] based on the provided 
        minimum and maximum values.

        Args:
            x: A 2D array of shape (N, 2) representing the input coordinates to be normalized.
            mins: A 1D array of shape (2,) representing the minimum values for each dimension.
            maxs: A 1D array of shape (2,) representing the maximum values for each dimension.

        Returns:
            A 2D array of shape (N, 2) containing the normalized coordinates in the range [0, 1].
        """
        deltas = maxs-mins
        delta = np.max(deltas)
        normed = (x - mins) / delta
        if deltas[0] > deltas[1]:
            normed[:,1] += (deltas[0]-deltas[1])/delta/2
        else:
            normed[:,0] += (deltas[1]-deltas[0])/delta/2
        return normed.astype(np.float32)

    @staticmethod
    def _smooth_contour(points: npt.NDArray[np.float32],
                        window_length: int,
                        polyorder: int) -> npt.NDArray[np.float32]:
        x = points[:,0]
        y = points[:,1]
        # wrap parameter (should be at least half the smoothing window)
        k = window_length // 2
        x = np.r_[x[-k:], x, x[:k]]
        y = np.r_[y[-k:], y, y[:k]]

        x_smooth = savgol_filter(x, window_length=window_length, polyorder=polyorder)[k:-k]
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)[k:-k]
        return np.vstack((x_smooth, y_smooth)).T

    @staticmethod
    def _resample_by_distance(points, spacing: int):
        """
        Resample a 2D point sequence to have approximately equal spacing between points.

        Parameters:
        - points: (N, 2) array of (x, y) coordinates.
        - spacing: desired distance between consecutive points.

        Returns:
        - resampled_points: (M, 2) array of resampled points.
        """
        points = np.asarray(points)
        if points.shape[0] < 2:
            return points.copy()

        # Compute cumulative distance along the polyline
        deltas = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))

        # Generate new equally spaced sample positions
        total_length = cumulative_lengths[-1]
        num_samples = max(int(np.floor(total_length / spacing)), 1)
        sample_distances = np.linspace(0, total_length, num_samples + 1)

        # Interpolate x and y separately
        x_interp = np.interp(sample_distances, cumulative_lengths, points[:, 0])
        y_interp = np.interp(sample_distances, cumulative_lengths, points[:, 1])

        resampled_points = np.vstack((x_interp, y_interp)).T
        return resampled_points


def get_neuromap(
        atlas: list[int],
        loading: list[float],
        shape: tuple[int],
        *,
        activity_threshold: float = Projection.DEFAULT_ACTIVITY_THRESHOLD,
        morph_beta: float = Morphing.DEFAULT_BETA,
        morph_z: float = Morphing.DEFAULT_Z,
        morph_sigma: float = Morphing.DEFAULT_SIGMA,
        projection_cluster_size_threshold: int = Projection.DEFAULT_CLUSTER_SIZE_THRESHOLD,
        projection_connectivity: Projection.ConnectivityType = Projection.DEFAULT_CONNECTIVITY,
        projection_sort_by_eigenvalue: bool = Projection.DEFAULT_SORT_BY_EIGENVALUE,
        contours_detection_scale: int = Contour.DEFAULT_DETECTION_SCALE,
        contours_spacing: int = Contour.DEFAULT_SPACING,
        contours_smoothing_window: int = Contour.DEFAULT_SMOOTHING_WINDOW,
        contours_smoothing_polynom: int = Contour.DEFAULT_SMOOTHING_POLYNOM,
        contours_cutoff_thresh: int = Contour.DEFAULT_CUTOFF_THRESH,
        contours_closing_kernel: tuple[int,int] = Contour.DEFAULT_CLOSING_KERNEL,
        saliency_grid_size: int = Saliency.DEFAULT_GRID_SIZE,
        ) -> tuple[
            npt.NDArray[np.float32],
            dict[int,list[npt.NDArray[np.float32]]],
            npt.NDArray[np.float32]
        ]:
    """
    Business logic for generating neuromaps from an atlas and loading values.

    Args:
        atlas: A list of integers representing the atlas labels for each voxel.
        loading: A list of floats representing the loading values for each voxel.
        shape: A tuple representing the dimensions of the 3D volume.
        activity_threshold: A float threshold for activity; values below this are ignored.
        morph_beta: A float parameter for morphing; controls the degree of morphing.
        morph_z: A float parameter for morphing; controls the z-axis scaling.
        morph_sigma: A float parameter for morphing; controls the radial expansion of the morph.
        projection_cluster_size_threshold: An integer threshold for cluster size in projection.
        projection_connectivity: A string indicating the type of connectivity for projection.
        projection_sort_by_eigenvalue: A boolean indicating whether to sort projection clusters by 
            eigenvalue. Generally, they should be sorted by eigenvalue to project along the dominant 
            directions of the clusters. However, as this was overlooked in the original 
            implementation this parameter is provided to allow for reproducing the original behavior 
            if needed.
        contours_detection_scale: An integer scale for contour detection.
        contours_spacing: An integer spacing for contour resampling.
        contours_smoothing_window: An integer window size for contour smoothing.
        contours_smoothing_polynom: An integer polynomial order for contour smoothing.
        contours_cutoff_thresh: An integer threshold for contour cutoff; contours with values below 
            this threshold are ignored.
        contours_closing_kernel: A tuple representing the kernel size for contour closing.
        saliency_grid_size: An integer grid size for saliency map generation.
    
    Raises:
        ValueError: If the lengths of atlas and loading do not match.
        ValueError: If the provided shape does not match the dimensions of atlas and loading.
        ValueError: If activity_threshold is not in the range [0, max(abs(loading))].
        ValueError: If morph_beta is not in the range [0, 1].
        ValueError: If morph_z is not greater than 0.
        ValueError: If morph_sigma is not greater than 0.
        ValueError: If saliency_grid_size is not a positive integer.
    
    Returns:
        A tuple containing:
        - saliency: A 2D array representing the saliency map with the specified grid size.
        - labels_contours: A dictionary mapping each label to its corresponding list of contour 
            arrays.
        - proj_vec: An array representing the projection vector. 
    """
    if len(atlas) != len(loading):
        raise ValueError("Atlas and loading size mismatch")
    if len(atlas) != np.prod(shape):
        raise ValueError("Given shape does not match the dimensions of atlas and loading")
    if not 0 <= activity_threshold <= max(np.abs(loading)):
        raise ValueError("activity_threshold must be larger 0 + smaller than max value of loading")
    if not 0 <= morph_beta <= 1:
        raise ValueError("morph_beta must be between 0 and 1")
    if 0 > morph_z:
        raise ValueError("morph_z must be larger than 0")
    if 0 > morph_sigma:
        raise ValueError("morph_sigma must be larger than 0")
    if 0 >= saliency_grid_size:
        raise ValueError("saliency_grid_size must be positive integer")

    # 1. morph
    locations = Morphing.get_morph(shape, morph_beta, morph_z, morph_sigma)

    # 2. projection
    proj_vec = Projection.project(atlas, loading, activity_threshold,
                                  locations, shape,
                                  projection_cluster_size_threshold,
                                  projection_connectivity,
                                  projection_sort_by_eigenvalue)

    proj_loc = locations@proj_vec.T
    brain_mask = np.array(atlas)>0
    norm_proj_loc = Contour.norm_space(proj_loc,
                                       np.min(proj_loc[brain_mask], axis=0),
                                       np.max(proj_loc[brain_mask], axis=0))
    label_positions = {
        l: norm_proj_loc[np.array(atlas)==l]
        for l in [label for label in np.unique(atlas) if label>0]
    }
    label_positions[0] = norm_proj_loc[brain_mask]

    # 3. contour outlines
    labels_contours = Contour.get_contours(
        label_positions,
        detection_scale=contours_detection_scale,
        spacing=contours_spacing,
        smoothing_window=contours_smoothing_window,
        smoothing_polynom=contours_smoothing_polynom,
        cutoff_thresh=contours_cutoff_thresh,
        closing_kernel=contours_closing_kernel)

    # 4. saliency maps
    saliency = Saliency.get_saliency_map(
        norm_proj_loc[brain_mask],
        np.array(loading)[brain_mask],
        grid_size=saliency_grid_size)

    return saliency, labels_contours, proj_vec
