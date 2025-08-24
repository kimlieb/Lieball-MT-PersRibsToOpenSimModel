function updated_locations = old_transform_pathpoints_relative_position(rib_data, transformations, varargin)
    % Transform path points using relative position method
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'visualize', false, @islogical);
    addParameter(p, 'debug', false, @islogical);
    parse(p, varargin{:});
    
    visualize = p.Results.visualize;
    debug_mode = p.Results.debug;
    updated_locations = [];
    
    fprintf('Transforming %d path points using relative position method...\n', length(rib_data));
    
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
        transform_data.debug = debug_mode; % Enable debug output if requested
        P_old = path_point.location;
        
        try
            % Apply relative position transformation
            P_new = apply_transformation_relative_position(P_old, transform_data);
            
            % Store updated location
            current_update.path_point_name = path_point.path_point_name;
            current_update.muscle_name = path_point.muscle_name;
            current_update.rib_name = path_point.rib_name;
            current_update.socket_frame = path_point.socket_frame;
            current_update.old_location = P_old;
            current_update.new_location = P_new;
            current_update.transformation_type = transform_data.type;
            current_update.movement_distance = norm(P_new - P_old) * 1000; % in mm
            
            % Add to array
            if isempty(updated_locations)
                updated_locations = current_update;
            else
                updated_locations(end+1) = current_update;
            end
            
            fprintf('Transformed %s (%s): [%.6f, %.6f, %.6f] -> [%.6f, %.6f, %.6f] (%.2f mm)\n', ...
                   path_point.path_point_name, transform_data.type, P_old, P_new, current_update.movement_distance);
            
        catch ME
            fprintf('Error transforming %s: %s\n', path_point.path_point_name, ME.message);
        end
    end
    
    if ~isempty(updated_locations)
        % Calculate transformation statistics
        movement_distances = [updated_locations.movement_distance];
        fprintf('\n=== TRANSFORMATION STATISTICS ===\n');
        fprintf('Successfully transformed: %d path points\n', length(updated_locations));
        fprintf('Average movement: %.2f mm\n', mean(movement_distances));
        fprintf('Max movement: %.2f mm\n', max(movement_distances));
        fprintf('Min movement: %.2f mm\n', min(movement_distances));
        fprintf('Std movement: %.2f mm\n', std(movement_distances));
    end
end