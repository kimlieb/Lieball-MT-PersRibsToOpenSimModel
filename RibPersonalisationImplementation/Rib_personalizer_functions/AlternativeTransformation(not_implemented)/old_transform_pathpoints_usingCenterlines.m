function updated_locations = old_transform_pathpoints_usingCenterlines(rib_data, transformations, varargin)
    % Transform path points using centerline-based transformations

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'visualize', false, @islogical);
    parse(p, varargin{:});

    visualize = p.Results.visualize;
    updated_locations = [];

    for i = 1:length(rib_data)
        path_point = rib_data(i);

        % Parse rib information
        rib_name_lower = lower(path_point.rib_name);
        rib_match = regexp(rib_name_lower, 'rib(\d+)_([lr])', 'tokens');

        if ~isempty(rib_match)
            rib_num = str2double(rib_match{1}{1});
            side = rib_match{1}{2};
            rib_key = sprintf('rib%d_%s', rib_num, side);
        else
            fprintf('Warning: Could not parse rib name: %s\n', path_point.rib_name);
            continue;
        end

        if ~isfield(transformations, rib_key)
            fprintf('Warning: No transformation found for %s\n', rib_key);
            continue;
        end

        transform_data = transformations.(rib_key);
        P_old = path_point.location;

        try
            % Apply linear blend transformation
            P_new = apply_transformation_fixed(P_old, transform_data);

            % Store updated location
            current_update.path_point_name = path_point.path_point_name;
            current_update.muscle_name = path_point.muscle_name;
            current_update.rib_name = path_point.rib_name;
            current_update.socket_frame = path_point.socket_frame;
            current_update.old_location = P_old;
            current_update.new_location = P_new;
            current_update.transformation_type = transform_data.type;

            % Add to array
            if isempty(updated_locations)
                updated_locations = current_update;
            else
                updated_locations(end+1) = current_update;
            end

            fprintf('Transformed %s (%s): [%.6f, %.6f, %.6f] -> [%.6f, %.6f, %.6f]\n', ...
                   path_point.path_point_name, transform_data.type, P_old, P_new);

        catch ME
            fprintf('Error transforming %s: %s\n', path_point.path_point_name, ME.message);
        end
    end

    fprintf('Successfully transformed %d path points\n', length(updated_locations));
end


% 
% function transform_data = create_centerline_transformation_linear_blend(old_centerline, new_centerline)
%     % Create transformation using linear interpolation along centerline
% 
%     fprintf('   Computing linear blend transformation...\n');
% 
%     old_vertices = old_centerline.vertices;
%     new_vertices = new_centerline.vertices;
% 
%     % Resample to same resolution if needed
%     if size(old_vertices, 1) ~= size(new_vertices, 1)
%         target_points = min(size(old_vertices, 1), size(new_vertices, 1));
%         old_vertices = resample_centerline(old_vertices, target_points);
%         new_vertices = resample_centerline(new_vertices, target_points);
%     end
% 
%     % Check and fix direction
%     [old_vertices, new_vertices] = check_and_fix_direction_simple(old_vertices, new_vertices);
% 
%     % Store transformation data for linear blend method
%     transform_data.type = 'linear_blend';
%     transform_data.old_centerline = old_vertices;
%     transform_data.new_centerline = new_vertices;
%     transform_data.old_length = compute_centerline_length(old_vertices);
%     transform_data.new_length = compute_centerline_length(new_vertices);
% 
%     fprintf('   Linear blend transformation computed successfully\n');
% end
% 
% function [old_vertices, new_vertices] = check_and_fix_direction_simple(old_vertices, new_vertices)
%     % Simple direction check and fix
% 
%     % Calculate total distance for current orientation
%     dist_normal = norm(old_vertices(1,:) - new_vertices(1,:)) + ...
%                   norm(old_vertices(end,:) - new_vertices(end,:));
% 
%     % Calculate total distance if we reverse new centerline
%     dist_reversed = norm(old_vertices(1,:) - new_vertices(end,:)) + ...
%                     norm(old_vertices(end,:) - new_vertices(1,:));
% 
%     % If reversed orientation gives better correspondence, reverse it
%     if dist_reversed < dist_normal
%         new_vertices = flipud(new_vertices);
%         fprintf('   Reversed new centerline for better correspondence\n');
%     end
% end
% 
% function total_length = compute_centerline_length(vertices)
%     % Compute total arc length of centerline
%     if size(vertices, 1) < 2
%         total_length = 0;
%         return;
%     end
%     distances = sqrt(sum(diff(vertices).^2, 2));
%     total_length = sum(distances);
% end
% 

% 


% 
% %%
% 
% 
% function P_new = apply_transformation(P_old, transform_data)
%     % Apply the appropriate transformation to a point
% 
%     switch transform_data.type
%         case 'tps'
%             P_new = apply_tps_transformation(P_old, transform_data);
%         case 'simple'
%             P_new = apply_simple_transformation(P_old, transform_data);
%         otherwise
%             error('Unknown transformation type: %s', transform_data.type);
%     end
% end
% 

% 









