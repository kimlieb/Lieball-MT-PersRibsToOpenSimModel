function rib_locations = extract_rib_pathpoints(osim_filename)
    % Extract location coordinates of muscle path points attached to ribs
    % Now uses OpenSim API to match the logic in update_osim_file.m
    
    try
        % Import OpenSim libraries
        import org.opensim.modeling.*
        
        % Load the OpenSim model
        fprintf('Loading OpenSim model for extraction: %s\n', osim_filename);
        model = Model(osim_filename);
        model.initSystem();
        state = model.initializeState();
        
        rib_locations = [];
        rib_count = 0;
        
        % Get all muscles
        muscles = model.getMuscles();
        fprintf('Scanning %d muscles for rib path points...\n', muscles.getSize());
        
        % Loop through all muscles
        for i = 0:muscles.getSize()-1
            muscle = muscles.get(i);
            muscle_name = char(muscle.getName());

            geom_path = muscle.getGeometryPath();
            path_point_set = geom_path.getPathPointSet();
            
            % Loop through all path points
            for j = 0:path_point_set.getSize()-1
                path_point_base = path_point_set.get(j);
                path_point = PathPoint.safeDownCast(path_point_base);
                path_point_name = char(path_point_base.getName());
                
                % Skip default path points (same as original)
                if startsWith(path_point_name, 'default*','IgnoreCase',true)
                    continue;
                end
                
                % Get parent frame
                parent_frame = path_point_base.getParentFrame();
                parent_body_name = char(parent_frame.getName());
                
                % Check if it's a rib body using the same logic as update_osim_file
                if is_rib_body(parent_body_name)
                    % Get current location
                    current_location = path_point.getLocation(state);
                    coords = [current_location.get(0), current_location.get(1), current_location.get(2)];
                    
                    rib_count = rib_count + 1;
                    
                    % Create struct for this path point
                    current_point.path_point_name = path_point_name;
                    current_point.muscle_name = muscle_name;
                    current_point.rib_name = parent_body_name; % Use the actual body name
                    current_point.socket_frame = sprintf('/bodyset/%s', parent_body_name); % Reconstruct socket frame format
                    current_point.location = coords;
                    current_point.x = coords(1);
                    current_point.y = coords(2);
                    current_point.z = coords(3);
                    
                    % Add to array
                    if isempty(rib_locations)
                        rib_locations = current_point;
                    else
                        rib_locations(end+1) = current_point;
                    end
                    
                    fprintf('Found rib path point: %s in muscle %s on %s\n', ...
                        path_point_name, muscle_name, parent_body_name);
                end
            end
        end
        
        fprintf('Total muscle rib path points found: %d\n', rib_count);
        
    catch ME
        fprintf('Error during extraction: %s\n', ME.message);
        fprintf('Falling back to XML parsing method...\n');
        
        % Fallback to original XML method if OpenSim API fails
        rib_locations = old_extract_rib_pathpoints(osim_filename);
    end
end


