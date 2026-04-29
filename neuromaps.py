from __future__ import annotations
from enum import StrEnum, IntEnum
import numpy.typing as npt
import numpy as np
from scipy.signal import savgol_filter
import cv2
import cc3d


class Projection:
    class ConnectivityType(StrEnum):
        LABEL = "label"
        CONNECTIVITY6 = "connectivity6"
        CONNECTIVITY18 = "connectivity18"
        CONNECTIVITY26 = "connectivity26"

    DEFAULT_ACTIVITY_THRESHOLD: float = 10
    DEFAULT_CLUSTER_SIZE_THRESHOLD: int = 200
    DEFAULT_CONNECTIVITY: ConnectivityType = ConnectivityType.LABEL

    # does not consider weighting based on loading -> very low loading equally threated as high one
    @staticmethod
    def project(
        atlas: list[int], 
        loading: list[float], 
        activity_threshold: float, 
        locations: npt.NDArray[np.float32], 
        shape: tuple[int], 
        cluster_size_threshold: int, 
        connectivity: Projection.ConnectivityType):
        mask: npt.NDArray[np.bool_] = np.abs(loading)>activity_threshold
        if not np.sum(mask):
            raise RuntimeError("Too high activity_threshold results in empty loading mask")

        clusters: dict[int, npt.NDArray[np.float32]] = Projection._get_projection_clusters(atlas, mask, locations, shape, cluster_size_threshold, connectivity)
        return Projection._calc_proj_vec(clusters)

    @staticmethod
    def _calc_proj_vec(Xn: dict[int, npt.NDArray[np.float32]]):
        X = np.vstack(list(Xn.values()))

        priori_probs = [len(x)/len(X) for x in Xn.values()]
        S_w = np.zeros((3,3))
        mu_l = {l:1/len(x)*np.sum(x, axis=0) for l,x in Xn.items()}
        for l,x in Xn.items():
            S_w += (x - np.array(len(x)*[mu_l[l]])).T@(x - np.array(len(x)*[mu_l[l]]))

        # between-class scatter matrix
        S_b = np.zeros((3,3))
        mus= list(mu_l.values())
        for i in range(len(Xn)-1):
            for j in range(i+1, len(Xn)):
                p_i = priori_probs[i]
                p_j = priori_probs[j]
                m_i = mus[i]
                m_j = mus[j]
                S_b += p_i*p_j*np.outer((m_i-m_j).T, (m_i-m_j))

        mu = 1/len(X)*np.sum(X, axis=0)
        S_b_ref = np.zeros((3,3))
        for l,x in Xn.items():
            S_b_ref += len(x)*np.outer((mu_l[l]-mu).T, (mu_l[l]-mu))

        _,eigvec = np.linalg.eig(np.linalg.inv(S_w)*S_b)
        return np.array([eigvec[0,:], eigvec[1,:]])

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
                components = {l: np.array(np.where(((atlas==l) & loading_mask).reshape(shape))).T for l in np.unique(atlas) if l > 0}
            case Projection.ConnectivityType.CONNECTIVITY6 | Projection.ConnectivityType.CONNECTIVITY18 | Projection.ConnectivityType.CONNECTIVITY26:
                mask = atlas.copy().astype(bool)
                mask[~loading_mask] = 0
                neighborhood_size: int = 6
                if connectivity==Projection.ConnectivityType.CONNECTIVITY18:
                    neighborhood_size = 18
                elif connectivity==Projection.ConnectivityType.CONNECTIVITY26:
                    neighborhood_size = 26
                labels: npt.NDArray[np.int32] = cc3d.connected_components(mask.reshape(shape), connectivity=neighborhood_size)
                components = {l: positions[(labels.flatten()==l) & loading_mask] for l in np.unique(atlas) if l > 0}
            case _:
                raise RuntimeError("Unsupported connectivity type")
        return {l: pos for l,pos in components.items() if len(pos)>cluster_size_threshold}

class Morphing:

    DEFAULT_BETA: float = .2
    DEFAULT_Z: float = 2
    DEFAULT_SIGMA: float = .2

    _VOLUME_ORIGIN: npt.NDArray[np.int32] = np.array([0, 0, -1])

    class Coords(IntEnum):
        X = 0
        Y = 1
        Z = 2

    @staticmethod
    def _get_normalized_positions(shape: tuple[int]):
        idx = np.indices(shape)
        idx = idx.reshape(idx.shape[Morphing.Coords.X], -1).T
        def map_pts(X):
            pts = 2*(1.0/shape[0]*np.array(X) - 0.5)
            pts[Morphing.Coords.Z] *= -1
            pts[Morphing.Coords.Z] += -1 - np.min(pts[2])
            return pts
        return map_pts(idx.T).T

    @staticmethod
    def get_morph(shape: tuple[int], beta: float, z_scale: float, sigma: float) -> npt.NDArray[np.float32]:
        P = Morphing._get_normalized_positions(shape)
        theta = np.arctan2((P[:,Morphing.Coords.X]**2+P[:,Morphing.Coords.Y]**2)**(1/2), P[:,Morphing.Coords.Z])
        
        P_proj = P-Morphing._VOLUME_ORIGIN
        psi_0 = np.arccos((P[:,Morphing.Coords.X]**2+P[:,Morphing.Coords.Y]**2)**(1/2) / (P_proj[:,Morphing.Coords.X]**2+P_proj[:,Morphing.Coords.Y]**2+P_proj[:,Morphing.Coords.Z]**2)**(1/2))
        psi_beta = -beta*psi_0 + psi_0
        r_0 = np.linalg.norm(P_proj, axis=1)
        r_e = np.pi-theta
        r_beta = beta*(r_e-r_0) + r_0

        P_z = np.tan(psi_beta) * np.linalg.norm(P[:,:Morphing.Coords.Z], axis=1)
        P_t_dir = np.hstack((P[:,:2], P_z[:,None]))
        P_t_norm = P_t_dir/np.linalg.norm(P_t_dir, axis=1)[:,None]
        P_t = Morphing._VOLUME_ORIGIN + r_beta[:,None]*P_t_norm

        # eviscerate movement in z-direction
        evisc_w_z = z_scale/2 * (1+P[:,Morphing.Coords.Z])
        evisc_w_xy = np.exp(-np.linalg.norm(P[:,:2], axis=1)**2/(2*sigma))
        P_z_max_evisc = evisc_w_xy * evisc_w_z
        P_t[:,Morphing.Coords.Z] += -beta*P_z_max_evisc
        return P_t

class Saliency:

    DEFAULT_GRID_SIZE: int = 40

    @staticmethod
    def get_saliency_map(locations: npt.NDArray[np.float32], weights: npt.NDArray[np.float32], grid_size: int) -> npt.NDArray[np.float32]:
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
    DEFAULT_DETECTION_SCALE: int = 80
    DEFAULT_SPACING: int = 2
    DEFAULT_SMOOTHING_WINDOW: int = 11
    DEFAULT_SMOOTHING_POLYNOM: int = 3
    DEFAULT_CUTOFF_THRESH: int = 20
    DEFAULT_CLOSING_KERNEL: tuple[int,int]=(3,3)
    @staticmethod
    def get_contours(locations: dict[int, npt.NDArray[np.float32]], detection_scale: int, spacing: int, smoothing_window: int, smoothing_polynom: int, cutoff_thresh: int, closing_kernel: tuple[int,int]) -> dict[int,list[npt.NDArray[np.float32]]]:

        for locs in locations.values():
            assert np.all(np.min(locs, axis=0)+np.finfo(np.float32).eps >= 0)
            assert np.all(np.max(locs, axis=0)-np.finfo(np.float32).eps <= 1)

        def contours_from_positions(pos: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
            label_idxs = np.ravel_multi_index((np.round(pos*(detection_scale-1))).astype(int).T, (detection_scale, detection_scale))
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
                    pruned_contours += [Contour._smooth_contour(resampled, smoothing_window, smoothing_polynom)]
            return pruned_contours

        labels_contours: dict[int,list[npt.NDArray[np.float32]]] = {}
        for l,xy in locations.items():
            labels_contours[l] = [c/detection_scale for c in contours_from_positions(xy)]
        return labels_contours

    @staticmethod
    def norm_space(X: npt.NDArray[np.float32], mins: npt.NDArray[np.float32], maxs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        deltas = maxs-mins
        delta = np.max(deltas)
        normed = (X - mins) / delta
        if deltas[0] > deltas[1]:
            normed[:,1] += (deltas[0]-deltas[1])/delta/2
        else:
            normed[:,0] += (deltas[1]-deltas[0])/delta/2
        return normed.astype(np.float32)

    @staticmethod
    def _smooth_contour(points: npt.NDArray[np.float32], window_length: int, polyorder: int) -> npt.NDArray[np.float32]:
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
        contours_detection_scale: int = Contour.DEFAULT_DETECTION_SCALE,
        contours_spacing: int = Contour.DEFAULT_SPACING,
        contours_smoothing_window: int = Contour.DEFAULT_SMOOTHING_WINDOW,
        contours_smoothing_polynom: int = Contour.DEFAULT_SMOOTHING_POLYNOM,
        contours_cutoff_thresh: int = Contour.DEFAULT_CUTOFF_THRESH,
        contours_closing_kernel: tuple[int,int] = Contour.DEFAULT_CLOSING_KERNEL,
        saliency_grid_size: int = Saliency.DEFAULT_GRID_SIZE,
        ) -> tuple[npt.NDArray[np.float32], dict[int,list[npt.NDArray[np.float32]]], npt.NDArray[np.float32]]:
    """
    tbd
    """
    if len(atlas) != len(loading):
        raise ValueError("Atlas and loading size mismatch")
    if len(atlas) != np.prod(shape):
        raise ValueError("Given shape does not match the dimensions of atlas and loading")
    if not 0 <= activity_threshold <= max(np.abs(loading)):
        raise ValueError("activity_threshold must be larger 0 and smaller than max value of loading")
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
    proj_vec = Projection.project(atlas, loading, activity_threshold, locations, shape, projection_cluster_size_threshold, projection_connectivity)

    proj_loc = locations@proj_vec.T
    brain_mask = np.array(atlas)>0
    norm_proj_loc = Contour.norm_space(proj_loc, np.min(proj_loc[brain_mask], axis=0), np.max(proj_loc[brain_mask], axis=0))
    label_positions = {l: norm_proj_loc[np.array(atlas)==l] for l in [label for label in np.unique(atlas) if label>0]}
    label_positions[0] = norm_proj_loc[brain_mask]

    # 3. contour outlines
    labels_contours = Contour.get_contours(label_positions, detection_scale=contours_detection_scale, spacing=contours_spacing, smoothing_window=contours_smoothing_window, smoothing_polynom=contours_smoothing_polynom, cutoff_thresh=contours_cutoff_thresh, closing_kernel=contours_closing_kernel)

    # 4. saliency maps
    saliency = Saliency.get_saliency_map(norm_proj_loc[brain_mask], np.array(loading)[brain_mask], grid_size=saliency_grid_size)

    return saliency, labels_contours, proj_vec
