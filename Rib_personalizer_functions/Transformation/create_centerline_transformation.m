function transform_data = create_centerline_transformation(old_centerline, new_centerline)
    % Create transformation using Thin Plate Spline (TPS) between centerlines
    
    fprintf('   Computing Thin Plate Spline transformation...\n');
    
    old_vertices = old_centerline.vertices;
    new_vertices = new_centerline.vertices;
    
    % RESOLUTION CONTROL - YOU CAN MODIFY THIS SECTION
    if size(old_vertices, 1) ~= size(new_vertices, 1)
        fprintf('   Old centerline: %d points, New centerline: %d points\n', ...
                size(old_vertices, 1), size(new_vertices, 1));
        fprintf('   Resampling centerlines to match point count...\n');
        
        % OPTION 1: Use minimum (current default)
        target_points = min(size(old_vertices, 1), size(new_vertices, 1));
        
        % OPTION 2: Use maximum (uncomment to use higher resolution)
        % target_points = max(size(old_vertices, 1), size(new_vertices, 1));
        
        % OPTION 3: Use fixed resolution (uncomment and set desired number)
        % target_points = 150;  % Use exactly 150 points for all transformations
        
        % OPTION 4: Use new centerline resolution (uncomment to always match new data)
        % target_points = size(new_vertices, 1);
        
        % OPTION 5: Use old centerline resolution (uncomment to always match old data)
        % target_points = size(old_vertices, 1);
        
        fprintf('   Target resolution: %d points\n', target_points);
        
        old_vertices = resample_centerline(old_vertices, target_points);
        new_vertices = resample_centerline(new_vertices, target_points);
    else
        fprintf('   Centerlines already have matching point count: %d points\n', size(old_vertices, 1));
    end
    
    % Compute TPS transformation
    try
        % Create TPS transformation matrices
        [tps_weights, tps_affine] = compute_tps_weights(old_vertices, new_vertices);
        
        % Store transformation data
        transform_data.type = 'tps';
        transform_data.source_points = old_vertices;
        transform_data.target_points = new_vertices;
        transform_data.tps_weights = tps_weights;
        transform_data.tps_affine = tps_affine;
        
        fprintf('   TPS transformation computed successfully\n');
        
    catch ME
        fprintf('   Warning: TPS failed, falling back to simple transformation\n');
        fprintf('   Error: %s\n', ME.message);
        
        % Fallback to simple transformation
        transform_data = create_simple_transformation(old_vertices, new_vertices);
    end
end

