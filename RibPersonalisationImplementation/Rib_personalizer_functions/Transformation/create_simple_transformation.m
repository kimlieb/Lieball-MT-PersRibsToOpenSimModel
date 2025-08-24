function transform_data = create_simple_transformation(old_vertices, new_vertices)
    % Create simple transformation as fallback
    
    fprintf('   Computing simple centroid + scaling transformation...\n');
    
    % Compute centroids
    old_centroid = mean(old_vertices, 1);
    new_centroid = mean(new_vertices, 1);
    
    % Compute bounding boxes for scaling
    old_bbox = max(old_vertices, [], 1) - min(old_vertices, [], 1);
    new_bbox = max(new_vertices, [], 1) - min(new_vertices, [], 1);
    
    % Avoid division by zero
    scale_factors = ones(1, 3);
    for i = 1:3
        if old_bbox(i) > 1e-6
            scale_factors(i) = new_bbox(i) / old_bbox(i);
        end
    end
    
    % Store transformation data
    transform_data.type = 'simple';
    transform_data.old_centroid = old_centroid;
    transform_data.new_centroid = new_centroid;
    transform_data.scale_factors = scale_factors;
    
    fprintf('   Simple transformation: scale=[%.3f %.3f %.3f]\n', scale_factors);
end
