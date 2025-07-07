import cv2
import numpy as np
import torch
from matplotlib import cm, colors
from numpy.ma import masked_where
from scipy.signal import peak_prominences, find_peaks
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn import cluster
from torch.nn import functional as F


# https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def create_circular_mask(h: int, w: int, center: (float, float) = None, radius: float = None, exclude_border=False):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center > radius
    # make sure that all borders are excluded
    if exclude_border:
        mask[:, 0] = True
        mask[0, :] = True
        mask[:, -1] = True
        mask[-1, :] = True
    return mask


def calculate_z_threshold_prominence(z_img: np.ndarray, masked: bool = False):
    def find_local_max(s):
        coordinate, peak_property = find_peaks(s, height=(None, None), prominence=(None, None), rel_height=(None, None))
        return coordinate, peak_property

    assert z_img.ndim == 2
    if not masked:
        z_img = np.ma.masked_where(create_circular_mask(*z_img.shape), z_img)
    x, x_peak_property = find_local_max(z_img.mean(axis=0))
    y, y_peak_property = find_local_max(z_img.mean(axis=1))
    return (x, y), (x_peak_property, y_peak_property)


def calculate_z_threshold(z_img: np.ndarray, masked: bool = False):
    def argmax_in_bound(x: np.ndarray, idx: int, radius: int):
        assert x.ndim == 1
        return np.argmax(x[max(0, idx - radius): min(len(x), idx + radius)])

    assert z_img.ndim == 2
    if not masked:
        z_img = np.ma.masked_where(create_circular_mask(*z_img.shape), z_img)

    (y, x) = np.unravel_index(np.argmax(z_img, axis=None), z_img.shape)

    prominence = (peak_prominences(z_img.mean(axis=0),
                                   [argmax_in_bound(z_img.mean(axis=0), x, int(z_img.shape[0] * 1))])[0],
                  peak_prominences(z_img.mean(axis=1),
                                   [argmax_in_bound(z_img.mean(axis=1), y, int(z_img.shape[1] * 1))])[0])
    return (x, y), prominence


class ZBlobThresholdExtractor:
    def __init__(self, spatial_dim: int, percentile: int = 5):
        """
        Uses a blob detector to calculate the threshold via the 8th percentile accept only quadratic images
        :param spatial_dim: int that defines height and width of the image
        """
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.minThreshold = 0.1
        detector_params.thresholdStep = 0.01

        detector_params.filterByArea = True
        detector_params.maxArea = spatial_dim ** 2 / 4

        self.detector = cv2.SimpleBlobDetector_create(detector_params)
        self.circular_mask = create_circular_mask(spatial_dim, spatial_dim)
        self.percentile = percentile

    def determine_z_threshold(self, img: np.ndarray, invert_img: bool = True):
        """

        :param img: image in range [0, 1]
        :param invert_img: invert image, because the blob detector search for black objects
        :return:
        """
        if img.ndim > 2:
            raise NotImplementedError("Only grayscaled images are supported.")
        assert img.shape[0] == img.shape[1]

        img_byte = (img * 255).astype(np.uint8)
        if invert_img:
            img_byte = cv2.bitwise_not(img_byte)
        kpts = self.detector.detect(img_byte)

        # process blob detections
        cover_by_blobs_mask = np.ones_like(img).astype(bool)
        for k in kpts:
            (x, y) = k.pt
            blob_diameter = k.size
            current_blob_mask = create_circular_mask(img.shape[0], img.shape[1],
                                                     center=(x, y),
                                                     radius=blob_diameter / 2)
            cover_by_blobs_mask = np.bitwise_and(cover_by_blobs_mask, current_blob_mask)
        cover_by_blobs_mask = np.bitwise_or(cover_by_blobs_mask, self.circular_mask)

        if not cover_by_blobs_mask.all():
            z_threshold = np.percentile(masked_where(cover_by_blobs_mask, img).compressed(), self.percentile)
            seg_mask = masked_where(img <= z_threshold, np.ones_like(cover_by_blobs_mask))
        else:
            z_threshold = float('nan')
            seg_mask = np.empty_like(img).fill(np.nan)

        return z_threshold, seg_mask, cover_by_blobs_mask


class GroundtruthExtractor:
    def __init__(self, mode: str, spatial_dim: int):
        self.extractor = ZBlobThresholdExtractor(spatial_dim)
        self.mode = mode

    def __call__(self, z: torch.Tensor):
        z = z.squeeze().numpy()
        if self.mode == 'seg':
            # blob + percentile for dilation
            _, seg_mask, _ = self.extractor.determine_z_threshold(z)
            seg_mask = torch.from_numpy(seg_mask.filled(False))
            return seg_mask
        elif self.mode == 'blob':
            _, _, blob_mask = self.extractor.determine_z_threshold(z)
            blob_mask = torch.from_numpy(~blob_mask)
            return blob_mask
        else:
            raise NotImplementedError(f"{self.mode} is not supported.")


def create_color_code(labels: np.ndarray, cmap_id: str = 'Set1'):
    assert labels.ndim == 1

    color_code = cm.get_cmap(cmap_id)(labels)
    color_code[labels == -1] = colors.to_rgba('dimgray', 1)

    return color_code


class ZSegmentationExtractor:
    def __init__(self, spatial_dim: int, avg_pool_kernel_size: int = 3, watershed_compactness: int = 1,
                 out_of_bounds_flag=-1, corner_margin: int = 10, edge_margin: int = 15, intensity_threshold: int = 30):
        """Initialize as before"""
        self.spatial_dim = spatial_dim
        self.avg_pool_kernel_size = avg_pool_kernel_size
        self.c = watershed_compactness
        self.bounds_flag = out_of_bounds_flag
        self.corner_margin = corner_margin
        self.edge_margin = edge_margin
        self.intensity_threshold = intensity_threshold
        
        # Initialize on CPU first
        self.device = torch.device('cpu')
        self.circular_mask = torch.from_numpy(~create_circular_mask(spatial_dim, spatial_dim, exclude_border=True))
        x = torch.linspace(0, spatial_dim-1, spatial_dim)
        y = torch.linspace(0, spatial_dim-1, spatial_dim)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')
        self.airways_label_linear_idx = torch.arange(spatial_dim ** 2).reshape((spatial_dim, spatial_dim))
        
        # Create edge mask
        self.edge_mask = np.ones((spatial_dim, spatial_dim), dtype=bool)
        self.edge_mask[:edge_margin, :] = False
        self.edge_mask[-edge_margin:, :] = False
        self.edge_mask[:, :edge_margin] = False
        self.edge_mask[:, -edge_margin:] = False

    def to(self, device):
        """Move tensors to device"""
        self.device = device
        self.circular_mask = self.circular_mask.to(device)
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        self.airways_label_linear_idx = self.airways_label_linear_idx.to(device)
        return self

    def extract_segmentation(self, z_img: torch.Tensor, rgb_img: np.ndarray = None, return_plot_data: bool = False):
        # Ensure z_img is on the correct device and get numpy version
        z_img = z_img.to(self.device)
        z_img_np = z_img.detach().cpu().numpy()
        
        assert z_img.dim() == 2
        assert z_img.shape[-2] == z_img.shape[-1] == self.spatial_dim

        # Create feature matrix
        F = torch.stack([self.grid_x, self.grid_y, z_img], dim=-1)
        mask_indices = self.circular_mask.bool()
        F = F[mask_indices]

        # Move to CPU for numpy operations
        F_cpu = F.detach().cpu().numpy()

        # Extract seed for kmeans
        z_min, z_max = np.min(F_cpu[:, 2]), np.max(F_cpu[:, 2])
        z_seed = np.array([[z_min], [z_max]])

        # KMeans clustering
        z_labels = cluster.KMeans(
            n_clusters=2, init=z_seed, n_init=1, tol=0.00001
        ).fit_predict(F_cpu[:, 2].reshape(-1, 1))
        is_airway = z_labels == 1

        # Consider only airways region
        F_airway_np = F_cpu[is_airway]
        
        # Convert threshold to numpy for comparison
        threshold_abs = np.min(F_airway_np[:, 2])
        
        # Extract local peaks as markers using numpy array
        initial_markers = peak_local_max(
            z_img_np,
            min_distance=round(self.spatial_dim * 0.02),
            threshold_abs=threshold_abs
        )

        # Filter out invalid points
        valid_markers = []
        for marker in initial_markers:
            y, x = int(marker[0]), int(marker[1])
            if self.is_valid_point(y, x, rgb_img):
                valid_markers.append(marker)

        valid_markers = np.array(valid_markers) if valid_markers else np.zeros((0, 2))

        # Sort peaks after xy-coordinates
        if len(valid_markers) > 0:
            coord_idx = np.lexsort((valid_markers[:, 0], valid_markers[:, 1]))
            valid_markers = valid_markers[coord_idx]

            # Convert markers to marker image
            airway_centroids_marker = np.zeros_like(z_img_np, dtype=int)
            for i, c_idx in enumerate(valid_markers):
                c_idx = (int(c_idx[0]), int(c_idx[1]))
                airway_centroids_marker[c_idx] = i + 1

            # Watershed segmentation
            airways_labels = watershed(
                -z_img_np,
                markers=airway_centroids_marker,
                mask=(z_img_np > threshold_abs),
                compactness=self.c
            )
        else:
            airways_labels = np.zeros_like(z_img_np)

        # Apply circular mask
        airways_labels[~self.circular_mask.cpu().numpy()] = self.bounds_flag
        
        # Convert results back to tensors and move to correct device
        seg_mask = torch.from_numpy(airways_labels).to(self.device)
        z_mask = torch.full_like(z_img, -1, dtype=torch.int, device=self.device)
        z_mask[self.circular_mask] = torch.from_numpy(z_labels).to(self.device)

        if not return_plot_data:
            return seg_mask.to(torch.int8), z_mask.to(torch.int8)
        else:
            # Convert F_airway back to tensor
            F_airway_tensor = torch.from_numpy(F_airway_np).to(self.device)
            return {
                'smoothed_z_img': z_img,
                'airway_centroids': torch.from_numpy(valid_markers).to(self.device),
                'F1': F_airway_tensor,
                'F': F,
                'z_labels': torch.from_numpy(z_labels).to(self.device),
                'airways_labels': torch.from_numpy(airways_labels).to(self.device),
                'is_airway': torch.from_numpy(is_airway).to(self.device),
                'seg_mask': seg_mask,
            }

    def is_valid_intensity(self, img: np.ndarray, y: int, x: int) -> bool:
        if img is None:
            return True
            
        if not (0 <= y < img.shape[0] and 0 <= x < img.shape[1]):
            return False
            
        pixel_values = img[y, x]
        
        if len(pixel_values.shape) > 0 and pixel_values.shape[0] == 3:
            gray_value = 0.299 * pixel_values[0] + 0.587 * pixel_values[1] + 0.114 * pixel_values[2]
        else:
            gray_value = pixel_values
            
        return not (np.abs(gray_value - 0) < self.intensity_threshold or 
                   np.abs(gray_value - 255) < self.intensity_threshold)
        
    def is_valid_point(self, y: int, x: int, img: np.ndarray = None) -> bool:
        if ((x < self.corner_margin and y < self.corner_margin) or
            (x < self.corner_margin and y > self.spatial_dim - self.corner_margin) or
            (x > self.spatial_dim - self.corner_margin and y < self.corner_margin) or
            (x > self.spatial_dim - self.corner_margin and y > self.spatial_dim - self.corner_margin)):
            return False
            
        if not self.edge_mask[y, x]:
            return False
            
        if not self.is_valid_intensity(img, y, x):
            return False
            
        return True

    def smooth_img(self, img: torch.Tensor):
        def box_filter(x, n):
            assert self.avg_pool_kernel_size % 2 == 1
            for _ in range(n):
                x = F.avg_pool2d(x, 
                               kernel_size=self.avg_pool_kernel_size,
                               padding=self.avg_pool_kernel_size // 2,
                               stride=1)
            return x

        img = img.view(1, 1, self.spatial_dim, self.spatial_dim)
        img = img.float()
        img = box_filter(img, 3)
        return img.squeeze()
