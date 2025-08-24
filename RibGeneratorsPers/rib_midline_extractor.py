import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from skimage import measure
from skimage.morphology import skeletonize, binary_closing, disk


class RibMidlineExtractor:
    def __init__(self, pa_segmentation_path, lateral_segmentation_path):
        """
        Initialize the rib midline extractor with biplanar segmentations

        Args:
            pa_segmentation_path: Path to the PA view segmentation (nifti file with multiple channels)
            lateral_segmentation_path: Path to the lateral view segmentation (nifti file with multiple channels)
        """
        # Step 1: Load segmentations
        pa_data = nib.load(pa_segmentation_path).get_fdata()
        lat_data = nib.load(lateral_segmentation_path).get_fdata()
        #lat_data = np.fliplr(lat_data)

        # Ensure segmentations are binary
        pa_binary = (pa_data > 0).astype(np.uint8)
        lat_binary = (lat_data > 0).astype(np.uint8)

        # Rotate both views by 90 degrees (clockwise)
        print("Rotating PA and LAT views by 90 degrees clockwise...")

        # Create new arrays with swapped dimensions to accommodate rotation
        # When rotating 90 degrees, height and width are swapped
        rotated_pa = np.zeros((pa_binary.shape[1], pa_binary.shape[0], pa_binary.shape[2]), dtype=pa_binary.dtype)
        rotated_lat = np.zeros((lat_binary.shape[1], lat_binary.shape[0], lat_binary.shape[2]), dtype=lat_binary.dtype)

        # Rotate each channel separately
        for i in range(pa_binary.shape[2]):
            # np.rot90 with k=3 rotates clockwise by 90 degrees
            # (k=3 is equivalent to rotating counterclockwise by 270 degrees)
            rotated_pa[:, :, i] = np.rot90(pa_binary[:, :, -(i+1)], k=4) # previously k=3 and just i, Kim changed because segmentation is flipped
            rotated_lat[:, :, i] = np.rot90(lat_binary[:, :, i], k=4)

        # Store the rotated arrays
        self.pa_seg = rotated_pa
        self.lat_seg = rotated_lat

        print(f"Original PA shape: {pa_binary.shape}, Rotated PA shape: {self.pa_seg.shape}")
        print(f"Original LAT shape: {lat_binary.shape}, Rotated LAT shape: {self.lat_seg.shape}")

        # Number of rib pairs
        print(f"PA shape: {self.pa_seg.shape}")
        print(f"LAT shape: {self.lat_seg.shape}")
        self.num_ribs = self.pa_seg.shape[2]

        # Print information about the segmentations
        print(f"PA segmentation: min={self.pa_seg.min()}, max={self.pa_seg.max()}, "
              f"unique values={np.unique(self.pa_seg)}")
        print(f"LAT segmentation: min={self.lat_seg.min()}, max={self.lat_seg.max()}, "
              f"unique values={np.unique(self.lat_seg)}")

        # Storage for midlines
        self.midlines = {}
        self.midlines_3d = {}

    def extract_midlines(self):
        """
        Extract midlines for all rib pairs in both views
        """
        for rib_idx in range(0, self.num_ribs): #previously 2 Kim changed
            print(f"Processing rib pair {rib_idx + 1}...")

            # Extract midlines for PA view
            pa_left, pa_right = self.extract_midlines_for_view(rib_idx, view='PA')

            # Extract midlines for lateral view
            lat_left, lat_right = self.extract_midlines_for_view(rib_idx, view='LAT')


            # Store midlines
            self.midlines[(rib_idx, 'PA', 'left')] = pa_left
            self.midlines[(rib_idx, 'PA', 'right')] = pa_right
            self.midlines[(rib_idx, 'LAT', 'left')] = lat_left
            self.midlines[(rib_idx, 'LAT', 'right')] = lat_right

    def extract_midlines_for_view(self, rib_idx, view='PA'):
        """
        Extract midlines for a rib pair in the specified view, with added
        morphological closing to improve mask quality before skeletonization

        Args:
            rib_idx: Index of the rib pair
            view: 'PA' or 'LAT'

        Returns:
            left_midline, right_midline: Arrays of points representing the midlines
        """
        # Get the segmentation mask for this rib
        seg = self.pa_seg[:, :, rib_idx] if view == 'PA' else self.lat_seg[:, :, rib_idx]

        # Ensure the segmentation is binary
        seg_binary = (seg > 0).astype(np.uint8)

        # For both PA and LAT views, use connected components to separate ribs
        labeled = measure.label(seg_binary)
        regions = measure.regionprops(labeled)

        # Create masks for left and right sides
        left_mask = np.zeros_like(seg)
        right_mask = np.zeros_like(seg)

        # SET UP MASKS # The assignment of left and right might be wrong depending on the way the EOS image is facing
        if len(regions) >= 2:
            # Sort regions by their centroid's x-coordinate
            regions.sort(key=lambda r: r.centroid[1])

            # Assign regions based on x-coordinate
            if len(regions) >= 1:
                left_mask[labeled == regions[1].label] = 1  # Assumes larger x- values -> right side
            if len(regions) >= 2:
                right_mask[labeled == regions[0].label] = 1  # Assumes Smaller x- values -> left side

            # Ensure masks are binary
            left_mask = (left_mask > 0).astype(np.uint8)
            right_mask = (right_mask > 0).astype(np.uint8)
        elif len(regions) == 1:
            # If we only have one component in the LAT view, it's likely that both left and right
            # ribs are merged in the segmentation. In this case, we'll use the same midline for both
            # left and right ribs
            if view == 'LAT':
                print(f"Found only one component in LAT view for rib {rib_idx}, using it for both sides")

                # Get the binary mask for the single component
                seg_binary = (labeled == regions[0].label)

                # Use the same mask for both left and right ribs
                left_mask = seg_binary
                right_mask = seg_binary
            else:
                # For other views with one region, assign it to both masks
                left_mask[labeled == regions[0].label] = 1
                right_mask[labeled == regions[0].label] = 1

        # Apply morphological closing operation to fill gaps and connect components
        # Define the structuring element (disk) for closing
        # Adjust the radius parameter (currently set to 2) based on your specific needs
        selem = disk(2)  # You can adjust the radius as needed

        # Apply binary closing to both masks
        left_mask_closed = binary_closing(left_mask, selem)
        right_mask_closed = binary_closing(right_mask, selem)

        # Print info about the effect of closing
        print(f"  Left mask before closing: {np.sum(left_mask)} pixels")
        print(f"  Left mask after closing: {np.sum(left_mask_closed)} pixels")
        print(f"  Right mask before closing: {np.sum(right_mask)} pixels")
        print(f"  Right mask after closing: {np.sum(right_mask_closed)} pixels")

        # Ensure masks are binary and of the right type for skeletonization
        # The skeletonize function works best with boolean arrays
        left_mask_binary = left_mask_closed.astype(bool)
        right_mask_binary = right_mask_closed.astype(bool)

        # SKELETONIZE
        # Print diagnostic information
        print(f"Rib {rib_idx}, {view} view:")
        print(f"  Left mask: shape={left_mask_binary.shape}, non-zero pixels={np.sum(left_mask_binary)}")
        print(f"  Right mask: shape={right_mask_binary.shape}, non-zero pixels={np.sum(right_mask_binary)}")

        # Skeletonize left mask
        left_skeleton = skeletonize(left_mask_binary, method='lee')
        print(f"  Left skeleton: non-zero pixels={np.sum(left_skeleton)}")
        # Store the skeleton directly - it's already a centerline
        left_midline = left_skeleton

        # Skeletonize right mask
        # For LAT view with one region, we're now using a flipped mask instead of copying
        right_skeleton = skeletonize(right_mask_binary, method='lee')
        print(f"  Right skeleton: non-zero pixels={np.sum(right_skeleton)}")
        # Store the skeleton directly - it's already a centerline
        right_midline = right_skeleton

        # If we're in LAT view with one region, print additional information
        if view == 'LAT' and len(regions) == 1:
            print(f"  Using horizontally flipped mask for right rib in lateral view")
            # The flipped mask should create a different skeleton for the right rib
            # This should help with the anatomical correspondence in 3D reconstruction

        return left_midline, right_midline

    def reconstruct_midlines_in_3d(self):
        """
        Reconstruct the 3D rib midlines with corrected left-right correspondence
        """
        # Create output directory for visualizations
        import os
        output_dir = "reconstruction_visualization"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # First, ensure we have a parametrizer with processed midlines
        from rib_midline_parametrization import RibMidlineParametrization
        parametrizer = RibMidlineParametrization(self)
        parametrizer.process_all_midlines()

        # Print debug info about parametric midlines
        print(f"After processing, found {len(parametrizer.parametric_midlines)} parametric midlines")

        # Count how many midlines we have for each view and side
        pa_left_count = sum(1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'PA' and k[2] == 'left')
        pa_right_count = sum(1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'PA' and k[2] == 'right')
        lat_left_count = sum(1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'LAT' and k[2] == 'left')
        lat_right_count = sum(1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'LAT' and k[2] == 'right')

        print(
            f"Midline counts - PA left: {pa_left_count}, PA right: {pa_right_count}, LAT left: {lat_left_count}, LAT right: {lat_right_count}")

        for rib_idx in range(0, self.num_ribs): #previously 2 Kim changed
            print(f"Reconstructing 3D midline for rib pair {rib_idx + 1}...")

            # Get parametric midlines from both views
            pa_left_key = (rib_idx, 'PA', 'left')
            pa_right_key = (rib_idx, 'PA', 'right')
            lat_left_key = (rib_idx, 'LAT', 'left')
            lat_right_key = (rib_idx, 'LAT', 'right')

            pa_left_param = parametrizer.parametric_midlines.get(pa_left_key)
            pa_right_param = parametrizer.parametric_midlines.get(pa_right_key)
            lat_left_param = parametrizer.parametric_midlines.get(lat_left_key)
            lat_right_param = parametrizer.parametric_midlines.get(lat_right_key)

            # Debug information about available midlines for this rib
            available = []
            if pa_left_param: available.append("PA left")
            if pa_right_param: available.append("PA right")
            if lat_left_param: available.append("LAT left")
            if lat_right_param: available.append("LAT right")
            print(f"  Available midlines for rib {rib_idx + 1}: {', '.join(available)}")

            # Try to reconstruct left rib
            if pa_left_param is not None and lat_left_param is not None:
                print(f"  Attempting to reconstruct left rib {rib_idx + 1}")

                # Since we've already corrected the direction in the parametrization step,
                # we can directly use the normal reconstruction without trying reversed parameters
                left_3d = self._direct_orthogonal_reconstruction(pa_left_param, lat_left_param)

                if left_3d is not None and len(left_3d) > 0:
                    self.midlines_3d[(rib_idx, 'left')] = left_3d
                    print(f"  Stored left rib with {len(left_3d)} points")
                else:
                    print("  Failed to reconstruct left rib")
            else:
                print(f"  Skipping left rib {rib_idx + 1} due to missing parametric midlines")

            # Right rib reconstruction
            if pa_right_param is not None and lat_right_param is not None:
                print(f"  Attempting to reconstruct right rib {rib_idx + 1}")

                # Since we've already corrected the direction in the parametrization step,
                # we can directly use the normal reconstruction without trying reversed parameters
                right_3d = self._direct_orthogonal_reconstruction(pa_right_param, lat_right_param)

                if right_3d is not None and len(right_3d) > 0:
                    self.midlines_3d[(rib_idx, 'right')] = right_3d
                    print(f"  Stored right rib with {len(right_3d)} points")
                else:
                    print("  Failed to reconstruct right rib")
            else:
                print(f"  Skipping right rib {rib_idx + 1} due to missing parametric midlines")

    def _direct_orthogonal_reconstruction(self, pa_param, lat_param):
        """
        Reconstruct a single rib with corrected coordinate system and parameterization
        """
        if pa_param is None or lat_param is None:
            print("  One of the parametric midlines is None")
            return None

        # Get sampled points from both parametric curves
        pa_points = pa_param.get('sampled_points')
        lat_points = lat_param.get('sampled_points')

        if pa_points is None or lat_points is None:
            print("  Missing sampled points in parametric midlines")
            return None

        # Resample both curves with the same number of points
        num_points = 100
        pa_t = np.linspace(0, 1, len(pa_points))
        lat_t = np.linspace(0, 1, len(lat_points))

        new_t = np.linspace(0, 1, num_points)

        pa_resampled = np.zeros((num_points, 2))
        lat_resampled = np.zeros((num_points, 2))

        for i in range(2):
            pa_resampled[:, i] = np.interp(new_t, pa_t, pa_points[:, i])
            # IMPORTANT: Consider inverting the parameter for lateral view
            # if curves are going in opposite directions
            lat_resampled[:, i] = np.interp(new_t, lat_t, lat_points[:, i])
            # Alternative: lat_resampled[:, i] = np.interp(new_t, lat_t[::-1], lat_points[::-1, i])

        # Create 3D points
        points_3d = np.zeros((num_points, 3))

        # CORRECTED COORDINATE MAPPING:
        # For synthetic orthogonal EOS, considering possible axis flipping
        for i in range(num_points):
            # Try these coordinate mappings:
            points_3d[i, 0] = pa_resampled[i, 1]  # x from PA view's x
            points_3d[i, 1] = pa_resampled[i, 0]  # y from PA view's y

            # IMPORTANT: This is where the z coordinate flipping might be needed
            # Try with and without the negative sign
            points_3d[i, 2] = lat_resampled[i, 1]  # z from LAT view's x
            # Alternative:
            #points_3d[i, 2] = -lat_resampled[i, 1]  # Flipped z coordinate

        return points_3d

    def _reconstruct_single_rib(self, pa_param, lat_param):
        """
        Reconstruct a single rib using parametric curve fitting with shape model refinement

        Args:
            pa_param: Parametric midline from PA view
            lat_param: Parametric midline from lateral view

        Returns:
            Array of 3D points representing the rib midline
        """
        if pa_param is None or lat_param is None:
            print("  One of the parametric midlines is None")
            return None

        # Debug: print the contents of the parametric midlines
        print(f"  PA parametric midline: {list(pa_param.keys())}")
        print(f"  LAT parametric midline: {list(lat_param.keys())}")

        # Get sampled points from both parametric curves
        pa_points = pa_param.get('sampled_points')
        lat_points = lat_param.get('sampled_points')

        if pa_points is None or lat_points is None:
            print("  Missing sampled points in parametric midlines")
            return None

        print(f"  PA points shape: {pa_points.shape if pa_points is not None else None}")
        print(f"  LAT points shape: {lat_points.shape if lat_points is not None else None}")

        # Make sure the arrays have the same length
        min_length = min(len(pa_points), len(lat_points))
        if min_length < len(pa_points) or min_length < len(lat_points):
            print(f"  Adjusting arrays to common length {min_length}")
            pa_points = pa_points[:min_length]
            lat_points = lat_points[:min_length]

        # Use the same parameter values t for both curves
        t_values = pa_param.get('t_params')
        if t_values is None or len(t_values) != len(pa_points):
            print(f"  Creating new t_values with length {len(pa_points)}")
            t_values = np.linspace(0, 1, len(pa_points))
        else:
            t_values = t_values[:min_length]

        # Create 3D points using proper triangulation
        points_3d = np.zeros((len(pa_points), 3))

        if self.epipolar is not None:
            print("  Using epipolar triangulation for 3D reconstruction")
            valid_points = 0

            for i in range(len(pa_points)):
                # Convert from (y, x) to (x, y) format for triangulation
                pa_xy = np.array([pa_points[i, 1], pa_points[i, 0]])
                lat_xy = np.array([lat_points[i, 1], lat_points[i, 0]])

                point_3d = self.epipolar.triangulate_point(pa_xy, lat_xy)

                if point_3d is not None and not np.any(np.isnan(point_3d)):
                    points_3d[i] = point_3d
                    valid_points += 1
                else:
                    # If triangulation fails, use the direct method as fallback
                    points_3d[i, 0] = pa_points[i, 1]  # x from PA
                    points_3d[i, 1] = (pa_points[i, 0] + lat_points[i, 0]) / 2.0  # Average y
                    points_3d[i, 2] = lat_points[i, 1]  # z from lateral

            print(f"  Successfully triangulated {valid_points} of {len(pa_points)} points")
        else:
            print("  No epipolar reconstructor available, using direct method")
            for i in range(len(pa_points)):
                # Use the direct method
                points_3d[i, 0] = pa_points[i, 1]  # x from PA
                points_3d[i, 1] = (pa_points[i, 0] + lat_points[i, 0]) / 2.0  # Average y
                points_3d[i, 2] = lat_points[i, 1]  # z from lateral

        print(f"  Created {len(points_3d)} 3D points")
        print(f"  3D points range: x={points_3d[:, 0].min():.1f} to {points_3d[:, 0].max():.1f}, "
              f"y={points_3d[:, 1].min():.1f} to {points_3d[:, 1].max():.1f}, "
              f"z={points_3d[:, 2].min():.1f} to {points_3d[:, 2].max():.1f}")

        # Apply shape model refinement (fit paraboloid)
        if len(points_3d) >= 4:  # Need at least 4 points for paraboloid fitting
            try:
                print("  Applying shape model refinement (paraboloid fitting)")
                # Fit a paraboloid of form z = ax^2 + by^2 + cxy + dx + ey + f
                A = np.column_stack([
                    points_3d[:, 0] ** 2, points_3d[:, 1] ** 2,
                    points_3d[:, 0] * points_3d[:, 1],
                    points_3d[:, 0], points_3d[:, 1],
                    np.ones(len(points_3d))
                ])

                # Solve for parameters using least squares
                params, residuals, rank, s = np.linalg.lstsq(A, points_3d[:, 2], rcond=None)
                print(f"  Paraboloid fitting results: rank={rank}, residuals={residuals}")

                # Create smooth paraboloid
                smooth_points = np.zeros_like(points_3d)
                smooth_points[:, 0] = points_3d[:, 0]  # Keep x coordinates
                smooth_points[:, 1] = points_3d[:, 1]  # Keep y coordinates

                # Compute z using paraboloid equation
                for i in range(len(points_3d)):
                    x, y = points_3d[i, 0], points_3d[i, 1]
                    z = (params[0] * x ** 2 + params[1] * y ** 2 + params[2] * x * y +
                         params[3] * x + params[4] * y + params[5])
                    smooth_points[i, 2] = z

                print(f"  Smoothed points range: z={smooth_points[:, 2].min():.1f} to {smooth_points[:, 2].max():.1f}")
                return smooth_points
            except Exception as e:
                print(f"  Failed to fit paraboloid: {e}")
                print("  Returning unsmoothed points instead")
                return points_3d
        else:
            print(f"  Not enough points ({len(points_3d)}) for shape model refinement")
            return points_3d

    def visualize_midlines(self, rib_indices=None):
        """
        Visualize the extracted midlines with two views:
        1. PA view (all ribs)
        2. LAT view (all ribs)

        Args:
            rib_indices: Indices of ribs to visualize, if None visualize all ribs
        """
        if not self.midlines:
            print("No midlines extracted yet. Run extract_midlines() first.")
            return

        # Create figure with black background - with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
        for ax in axes:
            ax.set_facecolor('black')

        # If no indices specified, use all ribs
        if rib_indices is None:
            rib_indices = range(0, self.num_ribs) #previously 2 Kim changed

        # Colors for different ribs
        colors = plt.cm.jet(np.linspace(0, 1, self.num_ribs))

        # Show segmentation as background
        pa_bg = np.sum(self.pa_seg, axis=2)
        pa_bg = pa_bg / np.max(pa_bg) if np.max(pa_bg) > 0 else pa_bg
        axes[0].imshow(pa_bg, cmap='gray', alpha=0.7)

        # Show LAT background
        lat_bg = np.sum(self.lat_seg, axis=2)
        lat_bg = lat_bg / np.max(lat_bg) if np.max(lat_bg) > 0 else lat_bg
        axes[1].imshow(lat_bg, cmap='gray', alpha=0.7)

        # Plot midlines for each rib
        for rib_idx in rib_indices:
            # PA view midlines
            pa_left = self.midlines.get((rib_idx, 'PA', 'left'))
            pa_right = self.midlines.get((rib_idx, 'PA', 'right'))

            # LAT view midlines
            lat_left = self.midlines.get((rib_idx, 'LAT', 'left'))
            lat_right = self.midlines.get((rib_idx, 'LAT', 'right'))

            # Plot PA view midlines (all ribs in one plot)
            if pa_left is not None:
                # Since we're storing the binary skeleton directly, we need to get coordinates
                y_coords, x_coords = np.where(pa_left)
                axes[0].scatter(x_coords, y_coords, s=1, color=colors[rib_idx],
                                label=f"Rib {rib_idx + 1} Left")
            if pa_right is not None:
                y_coords, x_coords = np.where(pa_right)
                axes[0].scatter(x_coords, y_coords, s=1, color=colors[rib_idx], marker='x',
                                label=f"Rib {rib_idx + 1} Right")

            # Plot LAT view midlines (both left and right in the same plot)
            if lat_left is not None:
                y_coords, x_coords = np.where(lat_left)
                axes[1].scatter(x_coords, y_coords, s=1, color=colors[rib_idx],
                                label=f"Rib {rib_idx + 1} Left")
                                
            if lat_right is not None:
                y_coords, x_coords = np.where(lat_right)
                axes[1].scatter(x_coords, y_coords, s=1, color=colors[rib_idx], marker='x',
                                label=f"Rib {rib_idx + 1} Right")

        # Set titles and labels
        axes[0].set_title("PA View - All Ribs", color='white')
        axes[0].set_xlabel("X", color='white')
        axes[0].set_ylabel("Y", color='white')
        axes[0].tick_params(colors='white')

        axes[1].set_title("Lateral View - All Ribs", color='white')
        axes[1].set_xlabel("X", color='white')
        axes[1].set_ylabel("Y", color='white')
        axes[1].tick_params(colors='white')

        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0),
                         ncol=min(6, len(handles)))
        for text in leg.get_texts():
            text.set_color('white')

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def visualize_all_rib_parametrizations(self):
        """
        Visualize all rib parametrizations (both left and right) in the same plot
        showing the matching points between PA and LAT views
        """
        # Import parametrizer if not already done
        from rib_midline_parametrization import RibMidlineParametrization

        # Create parametrizer if not already created
        parametrizer = RibMidlineParametrization(self)
        parametrizer.process_all_midlines()

        # Get all available ribs
        available_ribs = set()
        for key in parametrizer.parametric_midlines.keys():
            available_ribs.add(key[0])  # Add rib index

        available_ribs = sorted(list(available_ribs))

        if not available_ribs:
            print("No ribs available for visualization")
            return

        print(f"Available ribs: {available_ribs}")

        # Create figure with black background
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
        for ax in axes:
            ax.set_facecolor('black')

        # Show segmentation as background
        pa_bg = np.sum(self.pa_seg, axis=2)
        pa_bg = pa_bg / np.max(pa_bg) if np.max(pa_bg) > 0 else pa_bg
        axes[0].imshow(pa_bg, cmap='gray', alpha=0.7)

        lat_bg = np.sum(self.lat_seg, axis=2)
        lat_bg = lat_bg / np.max(lat_bg) if np.max(lat_bg) > 0 else lat_bg
        axes[1].imshow(lat_bg, cmap='gray', alpha=0.7)

        # Colors for different ribs
        colors = plt.cm.jet(np.linspace(0, 1, len(available_ribs)))

        # Plot all ribs
        legend_elements = []

        for i, rib_idx in enumerate(available_ribs):
            rib_color = colors[i]

            # Process both left and right sides
            for side in ['left', 'right']:
                # Get parametric midlines for the selected rib
                pa_key = (rib_idx, 'PA', side)
                lat_key = (rib_idx, 'LAT', side)

                pa_param = parametrizer.parametric_midlines.get(pa_key)
                lat_param = parametrizer.parametric_midlines.get(lat_key)

                if pa_param is None or lat_param is None:
                    continue

                # Get sampled points from both parametric curves
                pa_points = pa_param.get('sampled_points')
                lat_points = lat_param.get('sampled_points')

                if pa_points is None or lat_points is None:
                    continue

                # Use different marker styles for left and right
                marker = 'o' if side == 'left' else 'x'
                marker_size = 2
                alpha = 0.7

                # Plot the midlines
                axes[0].scatter(pa_points[:, 1], pa_points[:, 0], s=marker_size, color=rib_color,
                                marker=marker, alpha=alpha)
                axes[1].scatter(lat_points[:, 1], lat_points[:, 0], s=marker_size, color=rib_color,
                                marker=marker, alpha=alpha)

                # Mark start points with larger markers
                axes[0].scatter(pa_points[0, 1], pa_points[0, 0], s=50, color=rib_color,
                                marker='o', edgecolor='white', linewidth=1)
                # Make LAT start points more visible with a different marker and size
                axes[1].scatter(lat_points[0, 1], lat_points[0, 0], s=80, color=rib_color,
                                marker='*', edgecolor='white', linewidth=2)

                # Mark end points with larger markers
                axes[0].scatter(pa_points[-1, 1], pa_points[-1, 0], s=50, color=rib_color,
                                marker='s', edgecolor='white', linewidth=1)
                axes[1].scatter(lat_points[-1, 1], lat_points[-1, 0], s=50, color=rib_color,
                                marker='s', edgecolor='white', linewidth=1)

                # Add to legend only once
                if side == 'left':
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                      markerfacecolor=rib_color, markersize=8,
                                                      label=f"Rib {rib_idx + 1}"))

        # Add legend for start and end points
        pa_start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                                    markeredgecolor='white', markersize=8, label='PA Start Point')
        lat_start_point = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
                                     markeredgecolor='white', markersize=10, label='LAT Start Point')
        end_point = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                               markeredgecolor='white', markersize=8, label='End Point')
        left_side = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                               markersize=6, label='Left Rib')
        right_side = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray',
                                markersize=6, label='Right Rib')

        legend_elements.append(pa_start_point)
        legend_elements.append(lat_start_point)
        legend_elements.append(end_point)
        legend_elements.append(left_side)
        legend_elements.append(right_side)

        # Set titles and labels
        axes[0].set_title("PA View - All Ribs", color='white')
        axes[0].set_xlabel("X", color='white')
        axes[0].set_ylabel("Y", color='white')
        axes[0].tick_params(colors='white')

        axes[1].set_title("Lateral View - All Ribs", color='white')
        axes[1].set_xlabel("X", color='white')
        axes[1].set_ylabel("Y", color='white')
        axes[1].tick_params(colors='white')

        # Add legend
        leg = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0),
                         ncol=min(6, len(legend_elements)), frameon=False)
        for text in leg.get_texts():
            text.set_color('white')

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def visualize_3d_midlines(self, rib_indices=None, downsample_factor=1):
        """
        Enhanced visualization of the 3D reconstructed rib midlines with performance optimizations
        and improved depth ordering

        Args:
            rib_indices: Indices of ribs to visualize, if None visualize all ribs
            downsample_factor: Factor by which to downsample points for better performance
                              Higher values = fewer points = better performance but less detail
        """
        if not self.midlines_3d:
            print("No 3D midlines reconstructed yet. Run reconstruct_midlines_in_3d() first.")
            return

        # Create 3D figure with optimized renderer
        plt.rcParams['figure.figsize'] = [10, 8]  # Smaller figure size for better performance
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # If no indices specified, use all ribs
        if rib_indices is None:
            rib_indices = sorted(set([idx for idx, _ in self.midlines_3d.keys()]))

        # Colors for different ribs
        colors = plt.cm.jet(np.linspace(0, 1, self.num_ribs))

        # Track bounding box
        min_coords = np.array([float('inf'), float('inf'), float('inf')])
        max_coords = np.array([float('-inf'), float('-inf'), float('-inf')])

        # Create lists to store plot elements so we can control the rendering order
        plot_elements = []

        # First, collect all midlines and their data
        for rib_idx in rib_indices:
            left_3d = self.midlines_3d.get((rib_idx, 'left'))
            right_3d = self.midlines_3d.get((rib_idx, 'right'))

            if left_3d is not None and len(left_3d) > 0:
                # Downsample points for better performance
                if len(left_3d) > downsample_factor * 2:  # Only downsample if enough points
                    indices = np.linspace(0, len(left_3d) - 1, len(left_3d) // downsample_factor + 1).astype(int)
                    downsampled = left_3d[indices]
                else:
                    downsampled = left_3d

                # Always include start and end points
                if len(downsampled) > 0 and not np.array_equal(downsampled[0], left_3d[0]):
                    downsampled = np.vstack([left_3d[0], downsampled])
                if len(downsampled) > 0 and not np.array_equal(downsampled[-1], left_3d[-1]):
                    downsampled = np.vstack([downsampled, left_3d[-1]])

                # Calculate mean distance from camera/viewer
                # (approximate depth using y-coordinate for now)
                mean_depth = np.mean(downsampled[:, 1])

                # Store the line data and depth for later plotting
                plot_elements.append({
                    'type': 'line',
                    'data': downsampled,
                    'color': colors[rib_idx],
                    'style': '-',
                    'width': 2,
                    'label': f"Rib {rib_idx + 1} Left",
                    'depth': mean_depth,
                    'rib_idx': rib_idx,
                    'side': 'left'
                })

                # Store start/end points
                plot_elements.append({
                    'type': 'point',
                    'data': left_3d[0],
                    'color': 'green',
                    'size': 50,
                    'depth': left_3d[0, 1],
                    'rib_idx': rib_idx,
                    'side': 'left',
                    'position': 'start'
                })

                plot_elements.append({
                    'type': 'point',
                    'data': left_3d[-1],
                    'color': 'blue',
                    'size': 50,
                    'depth': left_3d[-1, 1],
                    'rib_idx': rib_idx,
                    'side': 'left',
                    'position': 'end'
                })

                # Update bounding box using full data
                min_coords = np.minimum(min_coords, np.min(left_3d, axis=0))
                max_coords = np.maximum(max_coords, np.max(left_3d, axis=0))

            if right_3d is not None and len(right_3d) > 0:
                # Downsample points for better performance
                if len(right_3d) > downsample_factor * 2:  # Only downsample if enough points
                    indices = np.linspace(0, len(right_3d) - 1, len(right_3d) // downsample_factor + 1).astype(int)
                    downsampled = right_3d[indices]
                else:
                    downsampled = right_3d

                # Always include start and end points
                if len(downsampled) > 0 and not np.array_equal(downsampled[0], right_3d[0]):
                    downsampled = np.vstack([right_3d[0], downsampled])
                if len(downsampled) > 0 and not np.array_equal(downsampled[-1], right_3d[-1]):
                    downsampled = np.vstack([downsampled, right_3d[-1]])

                # Calculate mean distance from camera/viewer
                mean_depth = np.mean(downsampled[:, 1])

                # Store the line data and depth for later plotting
                plot_elements.append({
                    'type': 'line',
                    'data': downsampled,
                    'color': colors[rib_idx],
                    'style': '-',
                    'width': 2,
                    'label': f"Rib {rib_idx + 1} Right",
                    'depth': mean_depth,
                    'rib_idx': rib_idx,
                    'side': 'right'
                })

                # Store start/end points
                plot_elements.append({
                    'type': 'point',
                    'data': right_3d[0],
                    'color': 'green',
                    'size': 50,
                    'depth': right_3d[0, 1],
                    'rib_idx': rib_idx,
                    'side': 'right',
                    'position': 'start'
                })

                plot_elements.append({
                    'type': 'point',
                    'data': right_3d[-1],
                    'color': 'red',
                    'size': 50,
                    'depth': right_3d[-1, 1],
                    'rib_idx': rib_idx,
                    'side': 'right',
                    'position': 'end'
                })

                # Update bounding box using full data
                min_coords = np.minimum(min_coords, np.min(right_3d, axis=0))
                max_coords = np.maximum(max_coords, np.max(right_3d, axis=0))

        # Sort plot elements by depth (back to front)
        # This ensures proper occlusion
        plot_elements.sort(key=lambda x: x['depth'], reverse=True)

        # Now plot the elements in the sorted order
        line_handles = []
        line_labels = []

        for element in plot_elements:
            if element['type'] == 'line':
                data = element['data']
                line, = ax.plot(data[:, 0], data[:, 1], data[:, 2],
                                element['style'], color=element['color'],
                                linewidth=element['width'], label=element['label'])

                # Only keep one handle per rib for the legend
                if (element['rib_idx'], element['side']) not in [(h['rib_idx'], h['side']) for h in line_handles]:
                    line_handles.append({
                        'handle': line,
                        'label': element['label'],
                        'rib_idx': element['rib_idx'],
                        'side': element['side']
                    })

            elif element['type'] == 'point':
                data = element['data']
                ax.scatter(data[0], data[1], data[2],
                           color=element['color'], s=element['size'])

        # Set axis labels with smaller font for better performance
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=8)  # Smaller tick labels

        # Simplified grid for better performance
        ax.grid(True, linestyle='--', alpha=0.3)

        # Simplified legend for better performance
        # Only show a subset of ribs in the legend to avoid clutter
        if line_handles:
            # Only show up to 6 ribs in the legend
            max_legend_items = min(6, len(line_handles))
            step = max(1, len(line_handles) // max_legend_items)

            handles = [h['handle'] for h in line_handles[::step]]
            labels = [h['label'] for h in line_handles[::step]]

            legend1 = ax.legend(handles, labels,
                                loc='upper right', fontsize=8, framealpha=0.5)
            ax.add_artist(legend1)

        # Add markers explanation with simplified style
        green_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8)
        red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8)
        ax.legend([green_dot, red_dot], ['Start', 'End'],
                  loc='upper left', fontsize=8, framealpha=0.5)

        # Set title with smaller font
        ax.set_title("3D Reconstructed Rib Midlines", fontsize=10)

        # Ensure the axis limits are not too tight
        range_x = max_coords[0] - min_coords[0]
        range_y = max_coords[1] - min_coords[1]
        range_z = max_coords[2] - min_coords[2]

        max_range = max(range_x, range_y, range_z) * 0.6

        mid_x = (max_coords[0] + min_coords[0]) / 2
        mid_y = (max_coords[1] + min_coords[1]) / 2
        mid_z = (max_coords[2] + min_coords[2]) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Ensure proper 3D visualization
        # Set the view angle to correctly show depth
        ax.view_init(elev=20, azim=-60)  # Adjust these values as needed

        # Simplified coordinate system axes
        origin = np.array([0, 0, 0])
        ax.quiver(origin[0], origin[1], origin[2],
                  30, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=1)
        ax.quiver(origin[0], origin[1], origin[2],
                  0, 30, 0, color='green', arrow_length_ratio=0.1, linewidth=1)
        ax.quiver(origin[0], origin[1], origin[2],
                  0, 0, 30, color='blue', arrow_length_ratio=0.1, linewidth=1)

        # Add text labels for axes with smaller font
        ax.text(30, 0, 0, "X", color='red', fontsize=8)
        ax.text(0, 30, 0, "Y", color='green', fontsize=8)
        ax.text(0, 0, 30, "Z", color='blue', fontsize=8)

        # Optimize figure layout
        plt.tight_layout()

        # Add a message about the downsampling
        plt.figtext(0.5, 0.01, f"Downsampled by factor of {downsample_factor} for better performance",
                    ha='center', fontsize=8)

        plt.show()

    #Kim added
    #scale_for_size = 2.54  # use 1 for mm, use 2.54 for px as in Galbusera code

    #def export_rib_endpoints_as_csv(self, output_path):
        #"""
        #Export rib start and end points to a .csv file.
        #Format: rib_index, side, start_x, start_y, start_z, end_x, end_y, end_z
        #"""
        #import csv

        #with open(output_path, mode="w", newline="") as file:
            #writer = csv.writer(file)
            #writer.writerow(["rib_index", "side", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"])

            #for (rib_idx, side), points in self.midlines_3d.items():
                #if points is None or len(points) == 0:
                    #continue

                #start = points[0]
                #end = points[-1]
                #writer.writerow([rib_idx + 1, side.upper(), *start, *end])

        #print(f"Exported rib start/end points to: {output_path}")

    def export_3d_midlines_as_txt(self, output_path):
        """
        Export 3D rib midlines to a MATLAB/OpenSim-friendly CSV-style .txt file.
        Columns: rib_index, side, point_index, x, y, z
        """
        with open(output_path, "w") as f:
            f.write("rib_index,side,point_index,x,y,z\n")
            for (rib_idx, side), points in sorted(self.midlines_3d.items()):
                for i, pt in enumerate(points):
                    f.write(f"{rib_idx + 1},{side.upper()[0]},{i},-{pt[2]:.6f},-{pt[1]:.6f},-{pt[0]:.6f}\n")  # Save in same orientation as AIS Coordinates
        print(f"Exported 3D midlines to: {output_path}")

    #def export_endpoints_as_txt(self, output_path):
        #"""
        #Export rib start/end points in tabular format for MATLAB/OpenSim use.
        #Columns: rib_index, side, type, x, y, z
        #"""

        #with open(output_path, "w") as f:
            #f.write("rib_index,side,type,x,y,z\n")
            #for (rib_idx, side), points in sorted(self.midlines_3d.items()):
                #start, end = points[0], points[-1]
                #f.write(f"{rib_idx + 1},{side.upper()[0]},start,{start[0]:.6f},{start[1]:.6f},{start[2]:.6f}\n")
                #f.write(f"{rib_idx + 1},{side.upper()[0]},end,{end[0]:.6f},{end[1]:.6f},{end[2]:.6f}\n")
                #f.write(f"{rib_idx + 1},{side.upper()[0]},start,{start[2]:.6f},-{start[0]:.6f},{start[1]:.6f}\n") #Galbusera Coordinates
                #f.write(f"{rib_idx + 1},{side.upper()[0]},end,{end[2]:.6f},-{end[0]:.6f},{end[1]:.6f}\n") #Galbusera Coordinates
                #f.write(f"{rib_idx + 1},{side.upper()[0]},start,{start[2]* scale_for_size:.6f},-{start[0]* scale_for_size:.6f},{start[1]* scale_for_size:.6f}\n")  # Galbusera Coordinates in px size
                #f.write(
                    #f"{rib_idx + 1},{side.upper()[0]},end,{end[2]* scale_for_size:.6f},-{end[0]* scale_for_size:.6f},{end[1]* scale_for_size:.6f}\n")  # Galbusera Coordinates in px size
                #f.write(f"{rib_idx + 1},{side.upper()[0]},start,{start[1]:.6f},-{start[0]:.6f},{start[2]:.6f}\n")  # AIS Coordinates
                #f.write(f"{rib_idx + 1},{side.upper()[0]},end,{end[1]:.6f},-{end[0]:.6f},{end[2]:.6f}\n")  # AIS Coordinates
        #print(f"Exported rib endpoints to: {output_path}")







