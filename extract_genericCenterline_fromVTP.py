import numpy as np
import xml.etree.ElementTree as ET
import re


class VTPCenterlineReader:
    def __init__(self, vtp_file_path):
        """
        Read centerline data from VTP file.

        Args:
            vtp_file_path (str): Path to the VTP file containing centerline
        """
        self.points = []
        self.connectivity = []
        self.load_vtp_centerline(vtp_file_path)

    def load_vtp_centerline(self, file_path):
        """Load centerline points and connectivity from VTP file."""
        try:
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract points
            points_elem = root.find('.//Points/DataArray[@Name="Points"]')
            if points_elem is not None:
                points_text = points_elem.text.strip()
                # Parse coordinates
                coords = [float(x) for x in points_text.split()]
                # Reshape to Nx3 array
                self.points = np.array(coords).reshape(-1, 3)
                print(f"Loaded {len(self.points)} points")
            else:
                raise ValueError("No Points data found in VTP file")

            # Extract line connectivity
            connectivity_elem = root.find('.//Lines/DataArray[@Name="connectivity"]')
            if connectivity_elem is not None:
                connectivity_text = connectivity_elem.text.strip()
                self.connectivity = [int(x) for x in connectivity_text.split()]
                print(f"Loaded {len(self.connectivity)} connectivity indices")
            else:
                print("Warning: No line connectivity found")
                # Create sequential connectivity if missing
                self.connectivity = []
                for i in range(len(self.points) - 1):
                    self.connectivity.extend([i, i + 1])

        except Exception as e:
            print(f"Error loading VTP file: {e}")
            raise

    def get_ordered_centerline(self):
        """
        Extract the centerline as an ordered sequence of points.

        Returns:
            numpy.ndarray: Ordered centerline points
        """
        if not self.connectivity:
            # If no connectivity, assume points are already in order
            return self.points

        # Build connectivity graph
        connections = {}
        for i in range(0, len(self.connectivity), 2):
            p1, p2 = self.connectivity[i], self.connectivity[i + 1]

            if p1 not in connections:
                connections[p1] = []
            if p2 not in connections:
                connections[p2] = []

            connections[p1].append(p2)
            connections[p2].append(p1)

        # Find start point (point with only one connection, or first point)
        start_points = [p for p, neighbors in connections.items() if len(neighbors) == 1]

        if start_points:
            start_point = start_points[0]
        else:
            start_point = 0  # Fallback to first point

        # Trace the path
        ordered_points = []
        current = start_point
        visited = set()

        while current is not None:
            if current in visited:
                break

            ordered_points.append(self.points[current])
            visited.add(current)

            # Find next unvisited neighbor
            next_point = None
            if current in connections:
                for neighbor in connections[current]:
                    if neighbor not in visited:
                        next_point = neighbor
                        break

            current = next_point

        return np.array(ordered_points)

    def ensure_origin_ordering(self, centerline_points):
        """
        Ensure centerline starts from point closest to origin.

        Args:
            centerline_points: Array of 3D points

        Returns:
            Array with points ordered so closest to origin is first
        """
        if len(centerline_points) < 2:
            return centerline_points

        # Calculate distances to origin for first and last points
        dist_first = np.linalg.norm(centerline_points[0])
        dist_last = np.linalg.norm(centerline_points[-1])

        print(f"Distance to origin - First point: {dist_first:.4f}, Last point: {dist_last:.4f}")

        # If last point is closer to origin, reverse the array
        if dist_last < dist_first:
            centerline_points = np.flip(centerline_points, axis=0)
            print("Reversed centerline order - closest to origin is now first")
        else:
            print("Centerline order is correct - closest to origin is already first")

        return centerline_points

    def resample_centerline(self, centerline_points, num_points=400):
        """
        Resample centerline to have exactly num_points.

        Args:
            centerline_points: Input centerline points
            num_points: Desired number of output points

        Returns:
            Resampled centerline points
        """
        if len(centerline_points) <= num_points:
            return centerline_points

        # Calculate cumulative arc length
        distances = [0]
        for i in range(1, len(centerline_points)):
            dist = np.linalg.norm(centerline_points[i] - centerline_points[i - 1])
            distances.append(distances[-1] + dist)

        total_length = distances[-1]

        # Generate evenly spaced parameter values
        target_distances = np.linspace(0, total_length, num_points)

        # Interpolate positions at target distances
        resampled_points = []

        for target_dist in target_distances:
            # Find the segment containing this distance
            for i in range(len(distances) - 1):
                if distances[i] <= target_dist <= distances[i + 1]:
                    # Linear interpolation within the segment
                    if distances[i + 1] == distances[i]:
                        t = 0
                    else:
                        t = (target_dist - distances[i]) / (distances[i + 1] - distances[i])

                    interpolated_pos = (1 - t) * centerline_points[i] + t * centerline_points[i + 1]
                    resampled_points.append(interpolated_pos)
                    break
            else:
                # Handle edge case for the last point
                resampled_points.append(centerline_points[-1].copy())

        return np.array(resampled_points)

    def save_centerline_obj(self, centerline_points, output_path):
        """Save centerline points as an OBJ file for visualization."""
        with open(output_path, 'w') as f:
            # Write vertices
            for point in centerline_points:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")

            # Write lines connecting consecutive points
            for i in range(len(centerline_points) - 1):
                f.write(f"l {i + 1} {i + 2}\n")

    def get_centerline_info(self):
        """Get information about the centerline."""
        if len(self.points) == 0:
            return "No centerline data loaded"

        ordered_points = self.get_ordered_centerline()

        # Calculate total length
        total_length = 0
        for i in range(1, len(ordered_points)):
            total_length += np.linalg.norm(ordered_points[i] - ordered_points[i - 1])

        # Get bounding box
        min_coords = np.min(ordered_points, axis=0)
        max_coords = np.max(ordered_points, axis=0)

        info = f"""
Centerline Information:
- Number of points: {len(ordered_points)}
- Total length: {total_length:.6f} units
- Bounding box:
  X: [{min_coords[0]:.6f}, {max_coords[0]:.6f}]
  Y: [{min_coords[1]:.6f}, {max_coords[1]:.6f}]
  Z: [{min_coords[2]:.6f}, {max_coords[2]:.6f}]
- Start point: [{ordered_points[0][0]:.6f}, {ordered_points[0][1]:.6f}, {ordered_points[0][2]:.6f}]
- End point: [{ordered_points[-1][0]:.6f}, {ordered_points[-1][1]:.6f}, {ordered_points[-1][2]:.6f}]
- Distance from origin (start): {np.linalg.norm(ordered_points[0]):.6f}
- Distance from origin (end): {np.linalg.norm(ordered_points[-1]):.6f}
"""
        return info


def process_vtp_centerline(vtp_file_path, num_points=400, output_path=None, ensure_origin_first=True):
    """
    Process VTP centerline file and optionally resample and save.

    Args:
        vtp_file_path (str): Path to VTP centerline file
        num_points (int): Number of points to resample to (None to keep original)
        output_path (str): Optional output OBJ file path
        ensure_origin_first (bool): Whether to ensure closest point to origin is first

    Returns:
        numpy.ndarray: Processed centerline points
    """
    # Load VTP centerline
    reader = VTPCenterlineReader(vtp_file_path)

    # Get ordered centerline
    centerline = reader.get_ordered_centerline()

    # Print info
    print(reader.get_centerline_info())

    # Ensure origin ordering if requested
    if ensure_origin_first:
        centerline = reader.ensure_origin_ordering(centerline)

    # Resample if requested
    if num_points and len(centerline) != num_points:
        print(f"Resampling from {len(centerline)} to {num_points} points...")
        centerline = reader.resample_centerline(centerline, num_points)

    # Save if output path provided
    if output_path:
        reader.save_centerline_obj(centerline, output_path)
        print(f"Saved centerline to {output_path}")

    return centerline


def batch_process_vtp_centerlines(input_folder, output_folder, num_points=400, naming_style="original"):
    """
    Batch process multiple VTP centerline files with flexible naming options.

    Args:
        input_folder (str): Folder containing VTP files
        output_folder (str): Output folder for OBJ files
        num_points (int): Number of points to resample to
        naming_style (str): Naming convention - "original", "centerline_prefix", or "custom"
    """
    import os
    import glob

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Find all VTP files
    vtp_pattern = os.path.join(input_folder, "*.vtp")
    vtp_files = glob.glob(vtp_pattern)

    print(f"Found {len(vtp_files)} VTP files to process")

    # Sort files to ensure consistent processing order (Rib1L, Rib1R, Rib2L, etc.)
    vtp_files.sort(key=lambda x: (
        int(re.search(r'Rib(\d+)', os.path.basename(x)).group(1)) if re.search(r'Rib(\d+)',
                                                                               os.path.basename(x)) else 999,
        'L' in os.path.basename(x)  # L comes before R
    ))

    for vtp_file in vtp_files:
        try:
            # Create output filename based on naming style
            base_name = os.path.splitext(os.path.basename(vtp_file))[0]

            if naming_style == "original":
                # Keep original name: Rib1R.obj, Rib1L.obj
                output_file = os.path.join(output_folder, f"{base_name}.obj")
            elif naming_style == "centerline_prefix":
                # Add centerline prefix: centerline_Rib1R.obj
                output_file = os.path.join(output_folder, f"centerline_{base_name}.obj")
            elif naming_style == "custom":
                # Custom naming for rib files: Rib01R_centerline.obj, Rib01L_centerline.obj
                # Extract rib number and side
                match = re.match(r'Rib(\d+)([LR])', base_name)
                if match:
                    rib_num = int(match.group(1))
                    side = match.group(2)
                    output_file = os.path.join(output_folder, f"Rib{rib_num:02d}{side}_centerline.obj")
                else:
                    # Fallback to original naming if pattern doesn't match
                    output_file = os.path.join(output_folder, f"{base_name}.obj")
            else:
                # Default to original naming
                output_file = os.path.join(output_folder, f"{base_name}.obj")

            print(f"\nProcessing {vtp_file}...")
            print(f"Output: {output_file}")

            # Process centerline
            centerline = process_vtp_centerline(
                vtp_file_path=vtp_file,
                num_points=num_points,
                output_path=output_file,
                ensure_origin_first=True
            )

            print(f"✓ Successfully processed {base_name}")

        except Exception as e:
            print(f"✗ Failed to process {vtp_file}: {e}")


def batch_process_rib_centerlines(input_folder, output_folder, num_points=400):
    """
    Specialized function for processing rib VTP files with rib-specific handling.

    Args:
        input_folder (str): Folder containing rib VTP files (Rib1L.vtp, Rib1R.vtp, etc.)
        output_folder (str): Output folder for OBJ files
        num_points (int): Number of points to resample to
    """
    import os
    import glob

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Find all rib VTP files
    rib_pattern = os.path.join(input_folder, "Rib*.vtp")
    vtp_files = glob.glob(rib_pattern)

    if not vtp_files:
        print(f"No rib VTP files found in {input_folder}")
        return

    print(f"Found {len(vtp_files)} rib VTP files to process")

    # Sort files by rib number and side
    def sort_key(filename):
        basename = os.path.basename(filename)
        match = re.search(r'Rib(\d+)([LR])', basename)
        if match:
            rib_num = int(match.group(1))
            side = match.group(2)
            return (rib_num, side)  # L comes before R in string comparison
        return (999, 'Z')  # Put unmatched files at the end

    vtp_files.sort(key=sort_key)

    # Process each file
    for vtp_file in vtp_files:
        try:
            base_name = os.path.splitext(os.path.basename(vtp_file))[0]

            # Extract rib information
            match = re.match(r'Rib(\d+)([LR])', base_name)
            if match:
                rib_num = int(match.group(1))
                side = match.group(2)
                side_full = "Left" if side == "L" else "Right"

                print(f"\n{'=' * 50}")
                print(f"Processing Rib {rib_num} - {side_full} ({base_name})")
                print(f"{'=' * 50}")

                # Create output filename
                output_file = os.path.join(output_folder, f"centerline_{base_name}.obj")
            else:
                print(f"\nProcessing {base_name} (non-standard naming)")
                output_file = os.path.join(output_folder, f"centerline_{base_name}.obj")

            # Process centerline
            centerline = process_vtp_centerline(
                vtp_file_path=vtp_file,
                num_points=num_points,
                output_path=output_file,
                ensure_origin_first=True
            )

            print(f"✓ Successfully processed {base_name}")
            print(f"  Output saved to: {os.path.basename(output_file)}")

        except Exception as e:
            print(f"✗ Failed to process {vtp_file}: {e}")

    print(f"\n{'=' * 50}")
    print("Batch processing complete!")
    print(f"{'=' * 50}")


# Example usage
if __name__ == "__main__":
    # Use the specialized rib processing function
    batch_process_rib_centerlines(
        input_folder="C:/Users/kimli/OneDrive - Universitaet Bern/MasterThesis_Kim/08_MT_final/08_03_OpenSim_Original_SGM02/SGM02/model/geometry/",
        output_folder="C:/Users/kimli/OneDrive - Universitaet Bern/MasterThesis_Kim/08_MT_final/08_01_Rib_Generators/08_01_D_CenterlineFromGeneric/genericRibs_CenterlinesFromVTP",
        num_points=100
    )

    # Alternative: Use the general batch function with different naming styles
    # batch_process_vtp_centerlines(
    #     input_folder="your_input_folder",
    #     output_folder="your_output_folder",
    #     num_points=400,
    #     naming_style="original"  # Options: "original", "centerline_prefix", "custom"
    # )
