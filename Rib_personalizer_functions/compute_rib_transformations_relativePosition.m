% You would also need to modify your transformation computation function
% to use the relative position method instead of TPS or linear blend:

function transformations = compute_rib_transformations_relativePosition(newCenterlinesTxt, oldCenterlinesFolder)
    % Compute transformations using relative position method
    
    fprintf('Computing rib transformations using relative position method...\n');
    
    % Load new centerlines
    new_centerlines = load_new_centerlines(newCenterlinesTxt);
    transformations = struct();
    
    % Define rib numbers and sides
    rib_numbers = 1:12;
    sides = {'L', 'R'};
    
    for rib_num = rib_numbers
        for side_idx = 1:length(sides)
            side = sides{side_idx};
            rib_key = sprintf('rib%d_%s', rib_num, lower(side));
            
            % Load old centerline
            old_rib_filename = sprintf('centerline_Rib%d%s.obj', rib_num, side);
            old_file_path = fullfile(oldCenterlinesFolder, old_rib_filename);
            
            if exist(old_file_path, 'file') && isfield(new_centerlines, rib_key)
                try
                    fprintf('   Processing %s...\n', rib_key);
                    
                    % Load old centerline
                    old_centerline = load_old_centerline(old_file_path);
                    new_centerline = new_centerlines.(rib_key);
                    
                    % Create relative position transformation
                    transform_data = create_relative_position_transformation(old_centerline, new_centerline);
                    transformations.(rib_key) = transform_data;
                    
                    fprintf('   ✓ %s transformation computed\n', rib_key);
                    
                catch ME
                    fprintf('   ✗ Error processing %s: %s\n', rib_key, ME.message);
                end
            else
                if ~exist(old_file_path, 'file')
                    fprintf('   ✗ Old centerline file not found: %s\n', old_rib_filename);
                end
                if ~isfield(new_centerlines, rib_key)
                    fprintf('   ✗ New centerline not found for: %s\n', rib_key);
                end
            end
        end
    end
    
    fprintf('Completed relative position transformations\n');
end