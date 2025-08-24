
function updated_locations = transform_pathpoints_relativePosition(rib_data, transformations, varargin)
    % Transform path points using relative position method and update OSIM file directly
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'visualize', false, @islogical);
    addParameter(p, 'debug', false, @islogical);
    addParameter(p, 'osim_input', '', @ischar);  % Input OSIM file path
    addParameter(p, 'osim_output', '', @ischar); % Output OSIM file path
    parse(p, varargin{:});
    
    visualize = p.Results.visualize;
    debug_mode = p.Results.debug;
    osim_input = p.Results.osim_input;
    osim_output = p.Results.osim_output;
    updated_locations = [];
    
    % Check if OSIM file paths are provided
    if isempty(osim_input) || isempty(osim_output)
        error('Both osim_input and osim_output file paths must be provided');
    end
    
    fprintf('Transforming %d path points using relative position method...\n', length(rib_data));
    fprintf('Will update OSIM file: %s -> %s\n', osim_input, osim_output);
    
    % Load the XML document
    try
        xml_doc = xmlread(osim_input);
        fprintf('Successfully loaded OSIM XML file\n');
    catch ME
        error('Could not read the OSIM file: %s. Error: %s', osim_input, ME.message);
    end
    
    % Debug: Show available transformation keys
    if debug_mode
        fprintf('Available transformation keys:\n');
        transform_keys = fieldnames(transformations);
        for i = 1:length(transform_keys)
            fprintf('  %s\n', transform_keys{i});
        end
        fprintf('\n');
    end
    
    % Process each path point
    for i = 1:length(rib_data)
        path_point = rib_data(i);
        
        fprintf('Processing path point: %s (muscle: %s, rib: %s)\n', ...
            path_point.path_point_name, path_point.muscle_name, path_point.rib_name);
        
        % Parse rib information - IMPROVED PARSING
        rib_name = path_point.rib_name;
        rib_key = parse_rib_key(rib_name);
        
        if isempty(rib_key)
            fprintf('Warning: Could not parse rib name: %s\n', rib_name);
            continue;
        end
        
        fprintf('  Parsed rib key: %s\n', rib_key);
        
        % Check if transformation exists
        if ~isfield(transformations, rib_key)
            fprintf('Warning: No transformation found for %s\n', rib_key);
            
            % Debug: Try to find similar keys
            if debug_mode
                fprintf('  Available keys containing similar patterns:\n');
                transform_keys = fieldnames(transformations);
                for k = 1:length(transform_keys)
                    if contains(lower(transform_keys{k}), lower(rib_key(1:4))) % Check first 4 chars (e.g., 'rib5')
                        fprintf('    %s\n', transform_keys{k});
                    end
                end
            end
            continue;
        end
        
        transform_data = transformations.(rib_key);
        transform_data.debug = debug_mode; % Enable debug output if requested
        P_old = path_point.location;
        
        try
            % Apply relative position transformation
            P_new = apply_transformation_relative_position(P_old, transform_data);
            
            % Update the XML directly
            success = update_pathpoint_in_xml(xml_doc, path_point.muscle_name, ...
                path_point.path_point_name, P_new, debug_mode);
            
            if success
                % Store updated location info
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
                
                fprintf('  ✓ Transformed and updated %s (%s): [%.6f, %.6f, %.6f] -> [%.6f, %.6f, %.6f] (%.2f mm)\n', ...
                       path_point.path_point_name, transform_data.type, P_old, P_new, current_update.movement_distance);
            else
                fprintf('  ✗ Failed to update XML for %s\n', path_point.path_point_name);
            end
            
        catch ME
            fprintf('Error transforming %s: %s\n', path_point.path_point_name, ME.message);
            if debug_mode
                fprintf('Stack trace:\n');
                for k = 1:length(ME.stack)
                    fprintf('  %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
                end
            end
        end
        
        fprintf('\n');
    end
    
    % Save the updated XML document
    try
        xmlwrite(osim_output, xml_doc);
        fprintf('Successfully saved updated OSIM file: %s\n', osim_output);
    catch ME
        error('Failed to save updated OSIM file: %s. Error: %s', osim_output, ME.message);
    end
    
    if ~isempty(updated_locations)
        % Calculate transformation statistics
        movement_distances = [updated_locations.movement_distance];
        fprintf('=== TRANSFORMATION STATISTICS ===\n');
        fprintf('Successfully transformed: %d path points\n', length(updated_locations));
        fprintf('Average movement: %.2f mm\n', mean(movement_distances));
        fprintf('Max movement: %.2f mm\n', max(movement_distances));
        fprintf('Min movement: %.2f mm\n', min(movement_distances));
        fprintf('Std movement: %.2f mm\n', std(movement_distances));
        
        % Show breakdown by rib
        unique_ribs = unique({updated_locations.rib_name});
        fprintf('\nBreakdown by rib:\n');
        for i = 1:length(unique_ribs)
            rib_points = updated_locations(strcmp({updated_locations.rib_name}, unique_ribs{i}));
            fprintf('  %s: %d path points\n', unique_ribs{i}, length(rib_points));
        end
    else
        fprintf('WARNING: No path points were successfully transformed!\n');
    end
end



