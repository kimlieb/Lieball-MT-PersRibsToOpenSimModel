function [old_centerlines, transformed_centerlines, personal_centerlines] = transform_all_centerlines(transformations, newCenterlinesTxt, oldCenterlinesFolder)
    % Transform all rib centerlines and organize data for visualization
    
    old_centerlines = struct();
    transformed_centerlines = struct();
    personal_centerlines = struct();
    
    % Load personal centerlines
    fprintf('   Loading personal centerlines...\n');
    personal_data = load_new_centerlines(newCenterlinesTxt);
    
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
            
            if exist(old_file_path, 'file')
                try
                    % Load old centerline
                    old_centerline = load_old_centerline(old_file_path);
                    old_centerlines.(rib_key) = old_centerline.vertices;
                    
                    % Get personal centerline
                    if isfield(personal_data, rib_key)
                        personal_centerlines.(rib_key) = personal_data.(rib_key).vertices;
                    end
                    
                    % Transform old centerline to new space
                    if isfield(transformations, rib_key)
                        transform_data = transformations.(rib_key);
                        transformed_points = transform_centerline_points(old_centerline.vertices, transform_data);
                        transformed_centerlines.(rib_key) = transformed_points;
                    end
                    
                catch ME
                    fprintf('   ✗ Error processing %s: %s\n', rib_key, ME.message);
                end
            end
        end
    end
end

function transformed_points = transform_centerline_points(centerline_points, transform_data)
    % Apply transformation to each point in the centerline
    
    transformed_points = zeros(size(centerline_points));
    
    for i = 1:size(centerline_points, 1)
        P_old = centerline_points(i, :);
        P_new = apply_transformation_fixed(P_old, transform_data);
        transformed_points(i, :) = P_new;
    end
end