function new_centerlines = load_new_centerlines(txt_file)
    % Load all new centerlines from the combined txt file
    
    new_centerlines = struct();
    
    % Read the CSV file
    data = readtable(txt_file);
    
    % Get unique rib combinations
    unique_ribs = unique(data(:, {'rib_index', 'side'}), 'rows');
    
    for i = 1:height(unique_ribs)
        rib_num = unique_ribs.rib_index(i);
        side = unique_ribs.side{i};
        
        % Filter data for this rib
        rib_mask = (data.rib_index == rib_num) & strcmp(data.side, side);
        rib_data = data(rib_mask, :);
        
        % Sort by point index to ensure correct order
        rib_data = sortrows(rib_data, 'point_index');
        
        % Extract coordinates
        vertices = [rib_data.x, rib_data.y, rib_data.z]*0.001;
        
        % Create centerline structure
        centerline.vertices = vertices;
        centerline.num_points = size(vertices, 1);
        
        % Store in structure
        rib_key = sprintf('rib%d_%s', rib_num, lower(side));
        new_centerlines.(rib_key) = centerline;
        
        fprintf('Loaded new centerline for %s: %d points\n', rib_key, centerline.num_points);
    end
end