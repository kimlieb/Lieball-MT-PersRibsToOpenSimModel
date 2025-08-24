def extract_centerline_medial_axis(self, num_points=100, smoothing_method='gaussian',
                                   filter_outliers=True, **smooth_params):
    """
    Extract centerline using a medial axis approximation approach.
    Enhanced with better smoothing options.
    """
    if self.vertex_normals is None:
        self.compute_vertex_normals()

    # Find approximate centerline points by looking for vertices that are
    # roughly equidistant from opposite sides of the tube
    centerline_candidates = []

    for i, vertex in enumerate(self.vertices):
        # Cast rays in multiple directions perpendicular to estimated tube direction
        # and find points that are roughly in the middle

        # Get neighboring vertices to estimate local tube direction
        if self.adjacency is None:
            self.build_adjacency_graph()

        neighbors = list(self.adjacency[i])
        if len(neighbors) < 2:
            continue

        # Estimate local direction as average of neighbor vectors
        neighbor_vectors = []
        for neighbor_idx in neighbors:
            vec = self.vertices[neighbor_idx] - vertex
            neighbor_vectors.append(vec / (np.linalg.norm(vec) + 1e-8))

        if len(neighbor_vectors) == 0:
            continue

        avg_direction = np.mean(neighbor_vectors, axis=0)
        avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-8)

        # Check if this vertex is roughly centered by examining distances
        # to vertices in perpendicular directions
        perpendicular_distances = []

        # Create perpendicular vectors
        if abs(avg_direction[2]) < 0.9:  # Not parallel to Z
            perp1 = np.cross(avg_direction, [0, 0, 1])
        else:  # Use Y if parallel to Z
            perp1 = np.cross(avg_direction, [0, 1, 0])

        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
        perp2 = np.cross(avg_direction, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)

        # Find nearby vertices in perpendicular directions
        threshold_distance = np.inf
        for neighbor_idx in neighbors:
            dist = np.linalg.norm(self.vertices[neighbor_idx] - vertex)
            threshold_distance = min(threshold_distance, dist * 3)  # Look within 3x nearest neighbor

        for direction in [perp1, -perp1, perp2, -perp2]:
            min_dist = np.inf
            for j, other_vertex in enumerate(self.vertices):
                if i == j:
                    continue

                to_other = other_vertex - vertex
                dist_along_perp = np.dot(to_other, direction)

                if dist_along_perp > 0:  # In the positive direction
                    lateral_component = to_other - dist_along_perp * direction
                    if np.linalg.norm(lateral_component) < threshold_distance:
                        min_dist = min(min_dist, dist_along_perp)

            if min_dist < np.inf:
                perpendicular_distances.append(min_dist)

        # If we have distances in multiple directions, check if they're balanced
        if len(perpendicular_distances) >= 2:
            dist_var = np.var(perpendicular_distances)
            avg_dist = np.mean(perpendicular_distances)

            # Lower variance indicates this vertex is more centered
            centerline_score = avg_dist / (1 + dist_var)
            centerline_candidates.append((centerline_score, i, vertex.copy()))

    if not centerline_candidates:
        # Fallback to simple path-based approach
        return self.extract_centerline_path_based(num_points, smoothing_method, filter_outliers)

    # Sort by centerline score and take the best candidates
    centerline_candidates.sort(reverse=True)

    # Take top candidates and sort them along the tube
    num_candidates = min(len(centerline_candidates), num_points * 2)
    selected_vertices = [candidate[2] for candidate in centerline_candidates[:num_candidates]]

    # Sort along the principal axis of the tube
    coords = np.array(selected_vertices)

    # Find principal axis using covariance
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    cov_matrix = np.cov(centered_coords.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Principal axis is the eigenvector with largest eigenvalue
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Project points onto principal axis and sort
    projections = np.dot(centered_coords, principal_axis)
    sorted_indices = np.argsort(projections)

    sorted_points = coords[sorted_indices]

    # Check if we need to reorder based on distance to origin
    dist_to_origin_first = np.linalg.norm(sorted_points[0])
    dist_to_origin_last = np.linalg.norm(sorted_points[-1])

    if dist_to_origin_last < dist_to_origin_first:
        # Reverse the order so closest to origin is first
        sorted_points = np.flip(sorted_points, axis=0)
        print(f"Reversed point order - closest to origin is now first")

    print(f"First point distance to origin: {np.linalg.norm(sorted_points[0]):.4f}")
    print(f"Last point distance to origin: {np.linalg.norm(sorted_points[-1]):.4f}")

    # Filter outliers if requested
    if filter_outliers:
        sorted_points = self.filter_outlier_points(sorted_points)

    # Apply smoothing based on method
    if smoothing_method == 'spline':
        final_points = self.fit_spline_centerline(sorted_points, num_points)
    else:
        # Apply smoothing then resample
        smoothed_points = self.smooth_path_positions(
            sorted_points,
            method=smoothing_method,
            **smooth_params
        )
        final_points = self.resample_path(smoothed_points, num_points)

    return final_points

import numpy as np
from collections import defaultdict
import heapq


class TubeCenterlineExtractor:
    def __init__(self, obj_file_path):
        """
        Initialize the extractor with an OBJ file containing the tube mesh.

        Args:
            obj_file_path (str): Path to the OBJ file
        """
        self.vertices = []
        self.faces = []
        self.load_obj(obj_file_path)
        self.vertex_normals = None
        self.adjacency = None

    def load_obj(self, file_path):
        """Load vertices and faces from OBJ file."""
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):
                    # Vertex
                    coords = line.split()[1:4]
                    vertex = [float(coord) for coord in coords]
                    self.vertices.append(vertex)
                elif line.startswith('f '):
                    # Face
                    face_data = line.split()[1:]
                    face = []
                    for vertex_data in face_data:
                        # Handle different OBJ face formats (v, v/vt, v/vt/vn)
                        vertex_idx = int(vertex_data.split('/')[0]) - 1  # OBJ is 1-indexed
                        face.append(vertex_idx)
                    self.faces.append(face)

        self.vertices = np.array(self.vertices)
        print(f"Loaded {len(self.vertices)} vertices and {len(self.faces)} faces")

    def compute_vertex_normals(self):
        """Compute vertex normals by averaging face normals."""
        vertex_normals = np.zeros_like(self.vertices)
        vertex_face_count = np.zeros(len(self.vertices))

        for face in self.faces:
            if len(face) >= 3:
                # Calculate face normal
                v1 = self.vertices[face[1]] - self.vertices[face[0]]
                v2 = self.vertices[face[2]] - self.vertices[face[0]]
                face_normal = np.cross(v1, v2)
                face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)

                # Add to vertex normals
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += face_normal
                    vertex_face_count[vertex_idx] += 1

        # Normalize vertex normals
        for i in range(len(vertex_normals)):
            if vertex_face_count[i] > 0:
                vertex_normals[i] /= vertex_face_count[i]
                vertex_normals[i] /= (np.linalg.norm(vertex_normals[i]) + 1e-8)

        self.vertex_normals = vertex_normals
        return vertex_normals

    def build_adjacency_graph(self):
        """Build vertex adjacency graph from faces."""
        adjacency = defaultdict(set)

        for face in self.faces:
            for i in range(len(face)):
                for j in range(i + 1, len(face)):
                    v1, v2 = face[i], face[j]
                    adjacency[v1].add(v2)
                    adjacency[v2].add(v1)

        self.adjacency = adjacency
        return adjacency

    def find_tube_ends(self):
        """
        Find the two ends of the tube by identifying vertices with lowest connectivity
        and maximum distance from each other.
        """
        if self.adjacency is None:
            self.build_adjacency_graph()

        # Find vertices with low connectivity (potential end points)
        connectivity = [(len(neighbors), idx) for idx, neighbors in self.adjacency.items()]
        connectivity.sort()

        # Consider the vertices with lowest 10% connectivity as potential endpoints
        num_candidates = max(10, len(connectivity) // 10)
        end_candidates = [idx for _, idx in connectivity[:num_candidates]]

        # Find the two most distant candidates
        max_distance = 0
        end1, end2 = end_candidates[0], end_candidates[1]

        for i in range(len(end_candidates)):
            for j in range(i + 1, len(end_candidates)):
                dist = np.linalg.norm(self.vertices[end_candidates[i]] - self.vertices[end_candidates[j]])
                if dist > max_distance:
                    max_distance = dist
                    end1, end2 = end_candidates[i], end_candidates[j]

        return end1, end2

    def dijkstra_path(self, start, end):
        """
        Find shortest path between two vertices using Dijkstra's algorithm.
        """
        if self.adjacency is None:
            self.build_adjacency_graph()

        distances = {vertex: float('inf') for vertex in self.adjacency}
        distances[start] = 0
        previous = {}
        pq = [(0, start)]
        visited = set()

        while pq:
            current_distance, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == end:
                break

            for neighbor in self.adjacency[current]:
                if neighbor in visited:
                    continue

                edge_weight = np.linalg.norm(self.vertices[current] - self.vertices[neighbor])
                distance = current_distance + edge_weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))

        # Reconstruct path
        path = []
        current = end
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()

        return path

    def smooth_path(self, path, smoothing_factor=0.5, iterations=5):
        """
        Smooth the path using iterative averaging.

        Args:
            path: List of vertex indices forming the path
            smoothing_factor: How much to smooth (0-1)
            iterations: Number of smoothing iterations
        """
        if len(path) < 3:
            return path

        smoothed_positions = [self.vertices[idx].copy() for idx in path]

        for _ in range(iterations):
            new_positions = [pos.copy() for pos in smoothed_positions]

            # Don't move the endpoints
            for i in range(1, len(smoothed_positions) - 1):
                # Average with neighbors
                neighbor_avg = (smoothed_positions[i - 1] + smoothed_positions[i + 1]) / 2
                new_positions[i] = (1 - smoothing_factor) * smoothed_positions[i] + smoothing_factor * neighbor_avg

            smoothed_positions = new_positions

        return smoothed_positions

    def smooth_path_gaussian(self, path, window_size=5, iterations=3):
        """
        Smooth path using Gaussian-like weighted averaging.
        Better for removing high-frequency noise while preserving overall shape.
        """
        if len(path) < 3:
            return path

        smoothed_positions = [self.vertices[idx].copy() for idx in path]

        # Create Gaussian-like weights
        half_window = window_size // 2
        weights = []
        for i in range(-half_window, half_window + 1):
            weight = np.exp(-0.5 * (i / (half_window / 2)) ** 2)
            weights.append(weight)
        weights = np.array(weights)
        weights /= np.sum(weights)

        for _ in range(iterations):
            new_positions = []

            for i in range(len(smoothed_positions)):
                if i < half_window or i >= len(smoothed_positions) - half_window:
                    # Keep endpoints and near-endpoints unchanged
                    new_positions.append(smoothed_positions[i].copy())
                else:
                    # Weighted average
                    weighted_sum = np.zeros(3)
                    for j, weight in enumerate(weights):
                        idx = i - half_window + j
                        weighted_sum += weight * smoothed_positions[idx]
                    new_positions.append(weighted_sum)

            smoothed_positions = new_positions

        return smoothed_positions

    def smooth_path_savgol(self, path, window_length=7, iterations=2):
        """
        Savitzky-Golay-like smoothing without scipy.
        Fits local polynomials to preserve trends while reducing noise.
        """
        if len(path) < window_length:
            return [self.vertices[idx].copy() for idx in path]

        smoothed_positions = [self.vertices[idx].copy() for idx in path]
        half_window = window_length // 2

        for _ in range(iterations):
            new_positions = []

            for i in range(len(smoothed_positions)):
                if i < half_window or i >= len(smoothed_positions) - half_window:
                    new_positions.append(smoothed_positions[i].copy())
                else:
                    # Fit a quadratic polynomial to local neighborhood
                    start_idx = i - half_window
                    end_idx = i + half_window + 1

                    # Extract local window
                    local_points = smoothed_positions[start_idx:end_idx]

                    # Simple polynomial smoothing: weighted average that emphasizes center
                    weights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1][:window_length])
                    weights = weights / np.sum(weights)

                    smoothed_point = np.zeros(3)
                    for j, point in enumerate(local_points):
                        smoothed_point += weights[j] * point

                    new_positions.append(smoothed_point)

            smoothed_positions = new_positions

        return smoothed_positions

    def fit_spline_centerline(self, path_positions, num_points=100, smoothing_strength=0.1):
        """
        Fit a smooth spline through the centerline points without scipy.
        Uses B-spline-like approach with control point reduction.
        """
        if len(path_positions) < 4:
            return self.resample_path(path_positions, num_points)

        positions = np.array(path_positions)

        # Reduce control points for smoother curve
        control_points = []
        step = max(1, len(positions) // (num_points // 4))  # Use fewer control points

        for i in range(0, len(positions), step):
            control_points.append(positions[i])

        # Ensure we include the last point
        if not np.array_equal(control_points[-1], positions[-1]):
            control_points.append(positions[-1])

        control_points = np.array(control_points)

        # Create smooth curve using Catmull-Rom-like interpolation
        smooth_points = []

        for i in range(len(control_points) - 1):
            # Get control points for interpolation
            p0 = control_points[max(0, i - 1)]
            p1 = control_points[i]
            p2 = control_points[i + 1]
            p3 = control_points[min(len(control_points) - 1, i + 2)]

            # Number of interpolation steps for this segment
            steps = max(1, num_points // len(control_points))

            for t in np.linspace(0, 1, steps, endpoint=(i == len(control_points) - 2)):
                # Catmull-Rom interpolation
                point = 0.5 * (
                        2 * p1 +
                        (-p0 + p2) * t +
                        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                        (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
                )
                smooth_points.append(point)

        # Resample to exact number of points
        return self.resample_path(smooth_points, num_points)

    def filter_outlier_points(self, path_positions, threshold_factor=2.0):
        """
        Remove points that deviate significantly from the local trend.
        """
        if len(path_positions) < 5:
            return path_positions

        positions = np.array(path_positions)
        filtered_positions = [positions[0]]  # Always keep first point

        for i in range(1, len(positions) - 1):
            # Check if current point is an outlier
            prev_vec = positions[i] - positions[i - 1]
            next_vec = positions[i + 1] - positions[i]

            # Expected position based on trend
            if i == 1:
                expected = positions[i - 1] + prev_vec
            else:
                # Use previous trend
                prev_trend = positions[i - 1] - positions[i - 2]
                expected = positions[i - 1] + prev_trend

            # Distance from expected position
            deviation = np.linalg.norm(positions[i] - expected)
            avg_step_size = (np.linalg.norm(prev_vec) + np.linalg.norm(next_vec)) / 2

            if avg_step_size > 0 and deviation / avg_step_size < threshold_factor:
                filtered_positions.append(positions[i])
            else:
                # Replace outlier with interpolated position
                filtered_positions.append((positions[i - 1] + positions[i + 1]) / 2)

        filtered_positions.append(positions[-1])  # Always keep last point
        return filtered_positions

    def resample_path(self, path_positions, num_points=100):
        """
        Resample the path to get exactly num_points evenly distributed along the curve.

        Args:
            path_positions: List of 3D positions along the path
            num_points: Number of points to extract
        """
        if len(path_positions) < 2:
            return path_positions

        # Calculate cumulative distances along the path
        distances = [0]
        for i in range(1, len(path_positions)):
            dist = np.linalg.norm(path_positions[i] - path_positions[i - 1])
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

                    interpolated_pos = (1 - t) * path_positions[i] + t * path_positions[i + 1]
                    resampled_points.append(interpolated_pos)
                    break
            else:
                # Handle edge case for the last point
                resampled_points.append(path_positions[-1].copy())

        return np.array(resampled_points)

    def extract_centerline_medial_axis(self, num_points=100):
        """
        Extract centerline using a medial axis approximation approach.
        This method works better for tubes with varying thickness.
        """
        if self.vertex_normals is None:
            self.compute_vertex_normals()

        # Find approximate centerline points by looking for vertices that are
        # roughly equidistant from opposite sides of the tube
        centerline_candidates = []

        for i, vertex in enumerate(self.vertices):
            # Cast rays in multiple directions perpendicular to estimated tube direction
            # and find points that are roughly in the middle

            # Get neighboring vertices to estimate local tube direction
            if self.adjacency is None:
                self.build_adjacency_graph()

            neighbors = list(self.adjacency[i])
            if len(neighbors) < 2:
                continue

            # Estimate local direction as average of neighbor vectors
            neighbor_vectors = []
            for neighbor_idx in neighbors:
                vec = self.vertices[neighbor_idx] - vertex
                neighbor_vectors.append(vec / (np.linalg.norm(vec) + 1e-8))

            if len(neighbor_vectors) == 0:
                continue

            avg_direction = np.mean(neighbor_vectors, axis=0)
            avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-8)

            # Check if this vertex is roughly centered by examining distances
            # to vertices in perpendicular directions
            perpendicular_distances = []

            # Create perpendicular vectors
            if abs(avg_direction[2]) < 0.9:  # Not parallel to Z
                perp1 = np.cross(avg_direction, [0, 0, 1])
            else:  # Use Y if parallel to Z
                perp1 = np.cross(avg_direction, [0, 1, 0])

            perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
            perp2 = np.cross(avg_direction, perp1)
            perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)

            # Find nearby vertices in perpendicular directions
            threshold_distance = np.inf
            for neighbor_idx in neighbors:
                dist = np.linalg.norm(self.vertices[neighbor_idx] - vertex)
                threshold_distance = min(threshold_distance, dist * 3)  # Look within 3x nearest neighbor

            for direction in [perp1, -perp1, perp2, -perp2]:
                min_dist = np.inf
                for j, other_vertex in enumerate(self.vertices):
                    if i == j:
                        continue

                    to_other = other_vertex - vertex
                    dist_along_perp = np.dot(to_other, direction)

                    if dist_along_perp > 0:  # In the positive direction
                        lateral_component = to_other - dist_along_perp * direction
                        if np.linalg.norm(lateral_component) < threshold_distance:
                            min_dist = min(min_dist, dist_along_perp)

                if min_dist < np.inf:
                    perpendicular_distances.append(min_dist)

            # If we have distances in multiple directions, check if they're balanced
            if len(perpendicular_distances) >= 2:
                dist_var = np.var(perpendicular_distances)
                avg_dist = np.mean(perpendicular_distances)

                # Lower variance indicates this vertex is more centered
                centerline_score = avg_dist / (1 + dist_var)
                centerline_candidates.append((centerline_score, i, vertex.copy()))

        if not centerline_candidates:
            # Fallback to simple path-based approach
            return self.extract_centerline_path_based(num_points)

        # Sort by centerline score and take the best candidates
        centerline_candidates.sort(reverse=True)

        # Take top candidates and sort them along the tube
        num_candidates = min(len(centerline_candidates), num_points * 2)
        selected_vertices = [candidate[2] for candidate in centerline_candidates[:num_candidates]]

        # Sort along the principal axis of the tube
        coords = np.array(selected_vertices)

        # Find principal axis using covariance
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        cov_matrix = np.cov(centered_coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Principal axis is the eigenvector with largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Project points onto principal axis and sort
        projections = np.dot(centered_coords, principal_axis)
        sorted_indices = np.argsort(projections)

        sorted_points = coords[sorted_indices]

        # Smooth and resample
        smoothed_points = self.smooth_path_positions(sorted_points)
        final_points = self.resample_path(smoothed_points, num_points)

        return final_points

    def smooth_path_positions(self, positions, method='gaussian', **kwargs):
        """
        Smooth a sequence of 3D positions using various methods.

        Args:
            positions: Array of 3D positions
            method: 'simple', 'gaussian', 'savgol', or 'spline'
            **kwargs: Method-specific parameters
        """
        if len(positions) < 3:
            return positions

        if method == 'simple':
            smoothing_factor = kwargs.get('smoothing_factor', 0.5)
            iterations = kwargs.get('iterations', 5)
            return self._smooth_simple(positions, smoothing_factor, iterations)
        elif method == 'gaussian':
            window_size = kwargs.get('window_size', 5)
            iterations = kwargs.get('iterations', 3)
            return self._smooth_gaussian(positions, window_size, iterations)
        elif method == 'savgol':
            window_length = kwargs.get('window_length', 7)
            iterations = kwargs.get('iterations', 2)
            return self._smooth_savgol(positions, window_length, iterations)
        elif method == 'spline':
            num_points = kwargs.get('num_points', len(positions))
            return self.fit_spline_centerline(positions, num_points)
        else:
            return positions

    def _smooth_simple(self, positions, smoothing_factor, iterations):
        """Simple averaging smoothing."""
        smoothed = np.array(positions)

        for _ in range(iterations):
            new_positions = smoothed.copy()

            for i in range(1, len(smoothed) - 1):
                neighbor_avg = (smoothed[i - 1] + smoothed[i + 1]) / 2
                new_positions[i] = (1 - smoothing_factor) * smoothed[i] + smoothing_factor * neighbor_avg

            smoothed = new_positions

        return smoothed

    def _smooth_gaussian(self, positions, window_size, iterations):
        """Gaussian-weighted smoothing."""
        smoothed = np.array(positions)
        half_window = window_size // 2

        # Create Gaussian weights
        weights = []
        for i in range(-half_window, half_window + 1):
            weight = np.exp(-0.5 * (i / (half_window / 2)) ** 2)
            weights.append(weight)
        weights = np.array(weights)
        weights /= np.sum(weights)

        for _ in range(iterations):
            new_positions = []

            for i in range(len(smoothed)):
                if i < half_window or i >= len(smoothed) - half_window:
                    new_positions.append(smoothed[i].copy())
                else:
                    weighted_sum = np.zeros(3)
                    for j, weight in enumerate(weights):
                        idx = i - half_window + j
                        weighted_sum += weight * smoothed[idx]
                    new_positions.append(weighted_sum)

            smoothed = np.array(new_positions)

        return smoothed

    def _smooth_savgol(self, positions, window_length, iterations):
        """Savitzky-Golay-like smoothing."""
        smoothed = np.array(positions)
        half_window = window_length // 2

        for _ in range(iterations):
            new_positions = []

            for i in range(len(smoothed)):
                if i < half_window or i >= len(smoothed) - half_window:
                    new_positions.append(smoothed[i].copy())
                else:
                    # Weighted average emphasizing center
                    weights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1][:window_length])
                    weights = weights / np.sum(weights)

                    smoothed_point = np.zeros(3)
                    start_idx = i - half_window
                    for j in range(window_length):
                        smoothed_point += weights[j] * smoothed[start_idx + j]

                    new_positions.append(smoothed_point)

            smoothed = np.array(new_positions)

        return smoothed

    def extract_centerline_path_based(self, num_points=100, smoothing_method='gaussian', filter_outliers=True):
        """
        Extract centerline by finding the shortest path between tube ends.
        Enhanced with better smoothing options.
        """
        # Find tube ends
        end1, end2 = self.find_tube_ends()
        print(f"Found tube ends: {end1} and {end2}")

        # Check which end is closer to origin
        dist_end1 = np.linalg.norm(self.vertices[end1])
        dist_end2 = np.linalg.norm(self.vertices[end2])

        if dist_end2 < dist_end1:
            # Swap so we start from the point closer to origin
            end1, end2 = end2, end1
            print(f"Swapped endpoints - starting from point closer to origin")

        # Find shortest path
        path = self.dijkstra_path(end1, end2)
        print(f"Found path with {len(path)} vertices")
        print(f"Start point distance to origin: {dist_end1:.4f}")
        print(f"End point distance to origin: {dist_end2:.4f}")

        # Convert to positions
        path_positions = [self.vertices[idx].copy() for idx in path]

        # Filter outliers if requested
        if filter_outliers:
            path_positions = self.filter_outlier_points(path_positions)
            print(f"Filtered path to {len(path_positions)} points")

        # Apply smoothing based on method
        if smoothing_method == 'spline':
            smoothed_positions = self.fit_spline_centerline(path_positions, num_points)
        else:
            # Apply smoothing
            smoothed_positions = self.smooth_path_positions(
                path_positions,
                method=smoothing_method,
                window_size=7,
                iterations=5,
                smoothing_factor=0.6
            )
            # Then resample
            smoothed_positions = self.resample_path(smoothed_positions, num_points)

        return smoothed_positions

    def extract_centerline(self, num_points=100, method='medial_axis', smoothing_method='gaussian',
                           filter_outliers=True, smoothing_strength='medium'):
        """
        Extract centerline points from the tube mesh with enhanced smoothing options.

        Args:
            num_points (int): Number of points to extract along the centerline
            method (str): 'medial_axis' or 'path_based'
            smoothing_method (str): 'simple', 'gaussian', 'savgol', or 'spline'
            filter_outliers (bool): Whether to remove outlier points
            smoothing_strength (str): 'light', 'medium', or 'heavy'

        Returns:
            numpy.ndarray: Array of shape (num_points, 3) containing centerline points
        """
        # Set smoothing parameters based on strength
        if smoothing_strength == 'light':
            smooth_params = {'iterations': 2, 'smoothing_factor': 0.3, 'window_size': 3}
        elif smoothing_strength == 'medium':
            smooth_params = {'iterations': 5, 'smoothing_factor': 0.5, 'window_size': 5}
        else:  # heavy
            smooth_params = {'iterations': 8, 'smoothing_factor': 0.7, 'window_size': 7}

        if method == 'medial_axis':
            final_points = self.extract_centerline_medial_axis(num_points, smoothing_method,
                                                       filter_outliers, **smooth_params)
        else:
            final_points = self.extract_centerline_path_based(num_points, smoothing_method, filter_outliers)
        # *** ALWAYS CHECK AND FIX ORDERING BASED ON DISTANCE TO ORIGIN ***
        final_points = self.ensure_origin_ordering(final_points)
        return final_points

    def save_centerline_obj(self, centerline_points, output_path):
        """Save centerline points as an OBJ file for visualization."""
        with open(output_path, 'w') as f:
            # Write vertices
            for point in centerline_points:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")

            # Write lines connecting consecutive points
            for i in range(len(centerline_points) - 1):
                f.write(f"l {i + 1} {i + 2}\n")

    def ensure_origin_ordering(self, centerline_points):
        """
        Ensure that the centerline starts from the point closest to origin (0,0,0).

        Args:
            centerline_points: Array of 3D points forming the centerline

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


# Example usage with different smoothing options
def extract_tube_centerline(obj_file_path, num_points=100, output_path=None,
                            smoothing_method='gaussian', smoothing_strength='medium',
                            filter_outliers=True):
    """
    Main function to extract centerline from a tube mesh with enhanced smoothing.

    Args:
        obj_file_path (str): Path to input OBJ file
        num_points (int): Number of centerline points to extract
        output_path (str): Optional path to save centerline as OBJ file
        smoothing_method (str): 'simple', 'gaussian', 'savgol', or 'spline'
        smoothing_strength (str): 'light', 'medium', or 'heavy'
        filter_outliers (bool): Remove outlier points that cause squiggles

    Returns:
        numpy.ndarray: Centerline points
    """
    extractor = TubeCenterlineExtractor(obj_file_path)

    # Try medial axis method first
    try:
        centerline_points = extractor.extract_centerline(
            num_points=num_points,
            method='medial_axis',
            smoothing_method=smoothing_method,
            smoothing_strength=smoothing_strength,
            filter_outliers=filter_outliers
        )
        print(f"Successfully extracted {len(centerline_points)} centerline points using medial axis method")
    except Exception as e:
        print(f"Medial axis method failed: {e}")
        print("Falling back to path-based method...")
        centerline_points = extractor.extract_centerline(
            num_points=num_points,
            method='path_based',
            smoothing_method=smoothing_method,
            smoothing_strength=smoothing_strength,
            filter_outliers=filter_outliers
        )
        print(f"Successfully extracted {len(centerline_points)} centerline points using path-based method")

    if output_path:
        extractor.save_centerline_obj(centerline_points, output_path)
        print(f"Saved centerline to {output_path}")

    return centerline_points





if __name__ == "__main__":

    # Generate all rib filenames (12 ribs, L and R sides)
    # Fixed: range should be 1 to 13 to include Rib12
    rib_obj_files = [f"Rib{i}{side}.obj" for i in range(1, 13) for side in ['L', 'R']]

    print("Processing rib files:")
    for filename in rib_obj_files:
        print(filename)

    print(f"\nTotal files: {len(rib_obj_files)}")

    # Base path for input files
    base_path = "C:/Users/kimli/OneDrive - Universitaet Bern/MasterThesis_Kim/08_MT_final/08_05_OpenSim_ModelAdapted/SGM02/model/geometry/"

    # Process all rib files
    for rib_file in rib_obj_files:
        try:
            # Create full input path
            input_path = base_path + rib_file

            # Create output filename
            rib_name = rib_file.replace('.obj', '')
            output_path = f"genericRibs_CenterlinesFromOBJ/centerline_{rib_name}.obj"

            print(f"\nProcessing {rib_file}...")

            # Extract centerline with heavy smoothing for ribs
            centerline = extract_tube_centerline(
                obj_file_path=input_path,
                num_points=400,
                smoothing_method='spline',  # spline is best for very noisy data
                smoothing_strength='heavy',
                filter_outliers=True,
                output_path=output_path
            )
            # *** ADD FINAL VERIFICATION ***
            first_point_dist = np.linalg.norm(centerline[0])
            last_point_dist = np.linalg.norm(centerline[-1])
            print(f"Final verification - First: {first_point_dist:.4f}, Last: {last_point_dist:.4f}")

            if first_point_dist > last_point_dist:
                print(f"WARNING: {rib_file} may have incorrect ordering!")

            print(f"✓ Successfully processed {rib_file} -> {output_path}")

        except Exception as e:
            print(f"✗ Failed to process {rib_file}: {str(e)}")
            continue

    print("\nBatch processing complete!")