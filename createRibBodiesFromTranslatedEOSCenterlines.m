clc; clear all;
%% Data
% Used Files
centerline_file = 'translated_centerlines_finalNewSegmentation.txt';

% Files which are created
rib_meshes_folderName = 'C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_01_Rib_Generators\08_01_C_OpenSim_RibGenerators\Output_RibMeshesNewSegmentation';

%% Main Function - Create Rib Bodies
function createRibBodiesfromCenterlines(dataFile, outputDir)
% CREATERIBBODIESINOPENSIM Creates rib bodies in OpenSim from centerline data

if nargin < 2
    outputDir = 'rib_meshes';
end

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Import OpenSim libraries
import org.opensim.modeling.*

%% Part A: Read and parse the data
fprintf('Reading rib data...\n');
data = readtable(dataFile);

% Get unique ribs
ribs = unique(data.rib_index);
sides = {'L', 'R'};

% Mesh density may influence further processing
rib_width = 0.008;   % Width in meters
rib_height = 0.012;  % Height in meters
n_cross_points = 16; % points around circumference
n_spline_points = 64; % points along length


%% Part B: Process each rib 
for rib_idx = ribs'
    for side_idx = 1:length(sides)
        side = sides{side_idx};
        
        fprintf('Processing Rib %d-%s...\n', rib_idx, side);
        
        % Extract data for this specific rib
        rib_data = data(data.rib_index == rib_idx & strcmp(data.side, side), :);
        
        if isempty(rib_data)
            continue;
        end
        
        % Sort by point index
        rib_data = sortrows(rib_data, 'point_index');
        
        % Extract coordinates
        points = [rib_data.x, rib_data.y, rib_data.z] * 0.001; % Convert to meters
        
        % VALIDATION: Check input data quality
        if size(points, 1) < 3
            fprintf('WARNING: Too few centerline points for Rib %d-%s, skipping\n', rib_idx, side);
            continue;
        end
        
        %% Create smooth spline through centerline points
        [spline_points, tangents, normals, binormals] = createRibSpline(points, n_spline_points);
        
        %% Generate 3D mesh geometry
        [vertices, faces] = generateRibMesh(spline_points, tangents, normals, binormals, ...
                                          rib_width, rib_height, n_cross_points);
        
        % VALIDATION: Check mesh quality
        fprintf('  Generated mesh: %d vertices, %d faces\n', size(vertices, 1), size(faces, 1));
        
        % Check for reasonable coordinate ranges
        x_range = max(vertices(:,1)) - min(vertices(:,1));
        y_range = max(vertices(:,2)) - min(vertices(:,2));
        z_range = max(vertices(:,3)) - min(vertices(:,3));
        
        fprintf('  Mesh size: %.3f × %.3f × %.3f meters\n', x_range, y_range, z_range);
        
        if max([x_range, y_range, z_range]) < 0.01 || max([x_range, y_range, z_range]) > 1.0
            fprintf('  WARNING: Unusual mesh dimensions detected!\n');
        end
        
        %% Calculate vertex normals for proper formats
        vertex_normals = calculateVertexNormals(vertices, faces);
        
        %% CRITICAL: Validate mesh before saving
        if validateMeshGeometry(vertices, faces, rib_idx, side)
            
            %% Save as VTP file (for CPD)
            vtp_filename = sprintf('Rib%d%s_pers.vtp', rib_idx, side);
            vtp_path = fullfile(outputDir, vtp_filename);
            writeVTP(vtp_path, vertices, faces);
            
            %% Save as OBJ file (for OpenSim)
            obj_filename = sprintf('Rib%d%s_pers.obj', rib_idx, side);
            obj_path = fullfile(outputDir, obj_filename);
            writeOpenSimOBJ(obj_path, vertices, faces, vertex_normals, rib_idx, side);
            
            %% OPTIONAL: Save as STL file (for verification)
            stl_filename = sprintf('Rib%d%s_pers.stl', rib_idx, side);
            stl_path = fullfile(outputDir, stl_filename);
            writeSTL(stl_path, vertices, faces);
            
            fprintf('  ✓ Created %s, %s, %s\n', vtp_filename, obj_filename, stl_filename);
        else
            fprintf('  ✗ Mesh validation failed, skipping file output\n');
        end
    end
end

fprintf('\n=== MESH GENERATION COMPLETE ===\n');
fprintf('Files created in: %s\n', outputDir);
fprintf('Mesh density: ~%d vertices per rib \n', n_spline_points * n_cross_points);
end

%% NEW: Mesh Validation Function
function isValid = validateMeshGeometry(vertices, faces, rib_idx, side)
% Validate mesh geometry for CPD compatibility

isValid = true;

% Check 1: Sufficient vertex count
if size(vertices, 1) < 100
    fprintf('    ERROR: Too few vertices (%d) for reliable CPD\n', size(vertices, 1));
    isValid = false;
end

% Check 2: Valid face connectivity
max_vertex_idx = max(faces(:));
if max_vertex_idx > size(vertices, 1)
    fprintf('    ERROR: Invalid face connectivity (references vertex %d, but only %d vertices exist)\n', ...
            max_vertex_idx, size(vertices, 1));
    isValid = false;
end

% Check 3: No degenerate faces
for i = 1:size(faces, 1)
    if length(unique(faces(i, :))) < 3
        fprintf('    ERROR: Degenerate face found at index %d\n', i);
        isValid = false;
        break;
    end
end

% Check 4: Reasonable coordinate ranges
coord_ranges = [range(vertices(:,1)), range(vertices(:,2)), range(vertices(:,3))];
if any(coord_ranges < 1e-6)
    fprintf('    ERROR: Degenerate geometry (range too small: [%.2e %.2e %.2e])\n', coord_ranges);
    isValid = false;
end

if any(coord_ranges > 2.0)
    fprintf('    WARNING: Very large geometry (range: [%.3f %.3f %.3f])\n', coord_ranges);
end

% Check 5: No NaN or Inf values
if any(isnan(vertices(:))) || any(isinf(vertices(:)))
    fprintf('    ERROR: NaN or Inf values in vertices\n');
    isValid = false;
end

if isValid
    fprintf('    ✓ Mesh validation passed\n');
else
    fprintf('    ✗ Mesh validation failed for Rib%d%s\n', rib_idx, side);
end
end

%%  Spline Creation Function with Error Handling
function [spline_points, tangents, normals, binormals] = createRibSpline(points, n_spline_points)
% Create smooth spline through rib centerline points with validation

% Validate input
if size(points, 1) < 3
    error('Need at least 3 points to create a spline');
end

% Remove duplicate points
[points, unique_idx] = unique(points, 'rows', 'stable');
if length(unique_idx) < size(points, 1)
    fprintf('    Removed %d duplicate points\n', size(points, 1) - length(unique_idx));
end

% Parameterize by cumulative distance
distances = [0; cumsum(sqrt(sum(diff(points).^2, 2)))];
total_length = distances(end);

if total_length < 1e-6
    error('Centerline points are too close together (total length: %.2e)', total_length);
end

% Create spline interpolation with error handling
try
    spline_x = spline(distances, points(:,1));
    spline_y = spline(distances, points(:,2));
    spline_z = spline(distances, points(:,3));
catch ME
    error('Spline creation failed: %s', ME.message);
end

% Generate points along spline
s_values = linspace(0, total_length, n_spline_points);
spline_points = zeros(n_spline_points, 3);
tangents = zeros(n_spline_points, 3);

for i = 1:n_spline_points
    s = s_values(i);
    
    % Position
    spline_points(i, 1) = ppval(spline_x, s);
    spline_points(i, 2) = ppval(spline_y, s);
    spline_points(i, 3) = ppval(spline_z, s);
    
    % Tangent (first derivative)
    tangents(i, 1) = ppval(fnder(spline_x), s);
    tangents(i, 2) = ppval(fnder(spline_y), s);
    tangents(i, 3) = ppval(fnder(spline_z), s);
end

% Normalize tangents with validation
for i = 1:n_spline_points
    tangent_norm = norm(tangents(i, :));
    if tangent_norm < 1e-8
        if i > 1
            tangents(i, :) = tangents(i-1, :);  % Use previous tangent
        else
            tangents(i, :) = [1, 0, 0];  % Default tangent
        end
    else
        tangents(i, :) = tangents(i, :) / tangent_norm;
    end
end

% Compute normal and binormal vectors using Frenet frame
normals = zeros(size(tangents));
binormals = zeros(size(tangents));

% Initial normal (more robust selection)
v = [0, 0, 1];
if abs(dot(tangents(1,:), v)) > 0.9
    v = [0, 1, 0];
    if abs(dot(tangents(1,:), v)) > 0.9
        v = [1, 0, 0];
    end
end

initial_cross = cross(tangents(1,:), v);
if norm(initial_cross) < 1e-8
    % Fallback for nearly parallel vectors
    initial_cross = [0, 1, 0];
end
normals(1,:) = initial_cross / norm(initial_cross);

% Propagate normal using parallel transport
for i = 2:n_spline_points
    prev_normal = normals(i-1,:);
    prev_tangent = tangents(i-1,:);
    curr_tangent = tangents(i,:);
    
    rotation_axis = cross(prev_tangent, curr_tangent);
    rotation_norm = norm(rotation_axis);
    
    if rotation_norm > 1e-6
        rotation_axis = rotation_axis / rotation_norm;
        cos_angle = dot(prev_tangent, curr_tangent);
        cos_angle = max(-1, min(1, cos_angle));  % Clamp to valid range
        angle = acos(cos_angle);
        
        % Rodrigues rotation formula
        normals(i,:) = prev_normal * cos(angle) + ...
                      cross(rotation_axis, prev_normal) * sin(angle) + ...
                      rotation_axis * dot(rotation_axis, prev_normal) * (1 - cos(angle));
    else
        normals(i,:) = prev_normal;
    end
    
    % Ensure normal is perpendicular to tangent
    normals(i,:) = normals(i,:) - dot(normals(i,:), curr_tangent) * curr_tangent;
    normal_norm = norm(normals(i,:));
    if normal_norm > 1e-8
        normals(i,:) = normals(i,:) / normal_norm;
    else
        % Fallback normal
        normals(i,:) = normals(i-1,:);
    end
end

% Compute binormals
for i = 1:n_spline_points
    binormals(i,:) = cross(tangents(i,:), normals(i,:));
    binormal_norm = norm(binormals(i,:));
    if binormal_norm > 1e-8
        binormals(i,:) = binormals(i,:) / binormal_norm;
    end
end
end

%% Mesh Generation Function
function [vertices, faces] = generateRibMesh(spline_points, tangents, normals, binormals, ...
                                           rib_width, rib_height, n_cross_points)

n_spline = size(spline_points, 1);
n_vertices = n_spline * n_cross_points;

vertices = zeros(n_vertices, 3);
faces = [];

% Generate cross-sectional profile (ellipse) - IMPROVED
theta = linspace(0, 2*pi, n_cross_points + 1);
theta = theta(1:end-1);  % Remove duplicate point

a = rib_width / 2;
b = rib_height / 2;

% Create elliptical cross-section
cross_section = [a * cos(theta)', b * sin(theta)'];

% Generate vertices
vertex_idx = 1;
for i = 1:n_spline
    center = spline_points(i, :);
    normal = normals(i, :);
    binormal = binormals(i, :);
    
    for j = 1:n_cross_points
        local_x = cross_section(j, 1);
        local_y = cross_section(j, 2);
        
        vertices(vertex_idx, :) = center + local_x * normal + local_y * binormal;
        vertex_idx = vertex_idx + 1;
    end
end

% Generate faces with proper winding order
for i = 1:n_spline-1
    for j = 1:n_cross_points
        % Current ring indices
        curr_base = (i-1) * n_cross_points;
        next_base = i * n_cross_points;
        
        % Vertex indices (1-based)
        v1 = curr_base + j;
        v2 = curr_base + mod(j, n_cross_points) + 1;
        v3 = next_base + j;
        v4 = next_base + mod(j, n_cross_points) + 1;
        
        % Create two triangles per quad with consistent winding
        faces = [faces; v1, v2, v3];  % First triangle
        faces = [faces; v2, v4, v3];  % Second triangle
    end
end

fprintf('    Generated %d vertices and %d faces\n', size(vertices, 1), size(faces, 1));
end

%% IMPROVED: Calculate Vertex Normals Function
function vertex_normals = calculateVertexNormals(vertices, faces)
% Calculate vertex normals by averaging face normals (area-weighted)
    n_vertices = size(vertices, 1);
    vertex_normals = zeros(n_vertices, 3);
    
    % Calculate face normals and accumulate at vertices
    for i = 1:size(faces, 1)
        v1_idx = faces(i, 1);
        v2_idx = faces(i, 2);
        v3_idx = faces(i, 3);
        
        v1 = vertices(v1_idx, :);
        v2 = vertices(v2_idx, :);
        v3 = vertices(v3_idx, :);
        
        % Calculate face normal
        edge1 = v2 - v1;
        edge2 = v3 - v1;
        face_normal = cross(edge1, edge2);
        
        % Area-weighted normal (length of cross product = 2 * area)
        face_area = norm(face_normal);
        if face_area > 1e-12
            face_normal = face_normal / face_area;  % Normalize
            % Weight by area for better normal computation
            face_normal = face_normal * face_area;
        else
            face_normal = [0, 0, 0];  % Degenerate face
        end
        
        % Accumulate at each vertex
        vertex_normals(v1_idx, :) = vertex_normals(v1_idx, :) + face_normal;
        vertex_normals(v2_idx, :) = vertex_normals(v2_idx, :) + face_normal;
        vertex_normals(v3_idx, :) = vertex_normals(v3_idx, :) + face_normal;
    end
    
    % Normalize vertex normals
    for i = 1:n_vertices
        normal_length = norm(vertex_normals(i, :));
        if normal_length > 1e-12
            vertex_normals(i, :) = vertex_normals(i, :) / normal_length;
        else
            vertex_normals(i, :) = [0, 0, 1];  % Default normal
        end
    end
end

%% Keep your existing file writing functions (STL, VTP, OBJ) - they're good!
function writeSTL(filename, vertices, faces)
fid = fopen(filename, 'w');
if fid == -1
    error('Cannot open file %s for writing', filename);
end

fprintf(fid, 'solid rib_mesh\n');

for i = 1:size(faces, 1)
    v1 = vertices(faces(i, 1), :);
    v2 = vertices(faces(i, 2), :);
    v3 = vertices(faces(i, 3), :);
    
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    normal = cross(edge1, edge2);
    normal_norm = norm(normal);
    if normal_norm > 1e-12
        normal = normal / normal_norm;
    else
        normal = [0, 0, 1];
    end
    
    fprintf(fid, '  facet normal %f %f %f\n', normal(1), normal(2), normal(3));
    fprintf(fid, '    outer loop\n');
    fprintf(fid, '      vertex %f %f %f\n', v1(1), v1(2), v1(3));
    fprintf(fid, '      vertex %f %f %f\n', v2(1), v2(2), v2(3));
    fprintf(fid, '      vertex %f %f %f\n', v3(1), v3(2), v3(3));
    fprintf(fid, '    endloop\n');
    fprintf(fid, '  endfacet\n');
end

fprintf(fid, 'endsolid rib_mesh\n');
fclose(fid);
end

function writeVTP(filename, vertices, faces)
% WRITEVTP Write mesh data to VTK XML PolyData format (.vtp)
% CRITICAL: This format is essential for CPD compatibility

fid = fopen(filename, 'w');
if fid == -1
    error('Cannot open file %s for writing', filename);
end

n_vertices = size(vertices, 1);
n_faces = size(faces, 1);

% Write XML header
fprintf(fid, '<?xml version="1.0"?>\n');
fprintf(fid, '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n');
fprintf(fid, '  <PolyData>\n');
fprintf(fid, '    <Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="%d">\n', n_vertices, n_faces);

% Write points - CRITICAL: Full precision for CPD
fprintf(fid, '      <Points>\n');
fprintf(fid, '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n');
for i = 1:n_vertices
    fprintf(fid, '          %.12f %.12f %.12f\n', vertices(i, 1), vertices(i, 2), vertices(i, 3));
end
fprintf(fid, '        </DataArray>\n');
fprintf(fid, '      </Points>\n');

% Write polygons
fprintf(fid, '      <Polys>\n');

% Connectivity (face indices, convert from 1-based to 0-based)
fprintf(fid, '        <DataArray type="Int32" Name="connectivity" format="ascii">\n');
for i = 1:n_faces
    fprintf(fid, '          %d %d %d\n', faces(i, 1)-1, faces(i, 2)-1, faces(i, 3)-1);
end
fprintf(fid, '        </DataArray>\n');

% Offsets (cumulative vertex count for each polygon)
fprintf(fid, '        <DataArray type="Int32" Name="offsets" format="ascii">\n');
for i = 1:n_faces
    fprintf(fid, '          %d\n', i * 3);
end
fprintf(fid, '        </DataArray>\n');

fprintf(fid, '      </Polys>\n');
fprintf(fid, '    </Piece>\n');
fprintf(fid, '  </PolyData>\n');
fprintf(fid, '</VTKFile>\n');

fclose(fid);
end

function writeOpenSimOBJ(filename, vertices, faces, vertex_normals, rib_index, side)
% WRITEOPENSIMOBJ Write mesh data to OpenSim-compatible Wavefront OBJ format
    fid = fopen(filename, 'w');
    if fid == -1
        error('Cannot open file %s for writing', filename);
    end
    
    % Write header
    fprintf(fid, '# Generated OBJ File: ''Rib%d%s_generated.blend''\n', rib_index, side);
    fprintf(fid, '# Generated from rib centerline data for CPD compatibility\n');
    fprintf(fid, '# Vertices: %d, Faces: %d\n', size(vertices, 1), size(faces, 1));
    
    % Write material library reference
    fprintf(fid, 'mtllib Rib%d%s.mtl\n', rib_index, side);
    
    % Write object name
    fprintf(fid, 'o Generated_Rib%d%s_RibCage.001\n', rib_index, side);
    
    % Write vertices with high precision
    for i = 1:size(vertices, 1)
        fprintf(fid, 'v %.12f %.12f %.12f\n', vertices(i, 1), vertices(i, 2), vertices(i, 3));
    end
    
    % Write vertex normals
    for i = 1:size(vertex_normals, 1)
        fprintf(fid, 'vn %.6f %.6f %.6f\n', vertex_normals(i, 1), vertex_normals(i, 2), vertex_normals(i, 3));
    end
    
    % Write material usage
    fprintf(fid, 'usemtl None.007\n');
    fprintf(fid, 's off\n');
    
    % Write faces with vertex//normal format
    for i = 1:size(faces, 1)
        fprintf(fid, 'f %d//%d %d//%d %d//%d\n', ...
                faces(i, 1), faces(i, 1), ...
                faces(i, 2), faces(i, 2), ...
                faces(i, 3), faces(i, 3));
    end
    
    fclose(fid);
end

%% Execution
% Step 1: Create ribs from centerline data
fprintf('===  Rib Mesh Generation ===\n');
createRibBodiesfromCenterlines(centerline_file, rib_meshes_folderName);