function rib_locations = old_extract_rib_pathpoints(osim_filename)
    % Extract location coordinates of muscle path points attached to ribs
    
    try
        xml_doc = xmlread(osim_filename);
    catch
        error('Could not read the OSIM file: %s', osim_filename);
    end
    
    rib_locations = [];
    rib_count = 0;
    
    % Get all muscle elements
    muscle_types = {'Millard2012EquilibriumMuscle'};
    
    for muscle_type_idx = 1:length(muscle_types)
        muscle_elements = xml_doc.getElementsByTagName(muscle_types{muscle_type_idx});
        
        for muscle_idx = 0:muscle_elements.getLength()-1
            muscle_element = muscle_elements.item(muscle_idx);
            muscle_name = char(muscle_element.getAttribute('name'));
            
            % Get all PathPoint elements within this muscle
            path_points = muscle_element.getElementsByTagName('PathPoint');
            
            for i = 0:path_points.getLength()-1
                path_point = path_points.item(i);
                path_point_name = char(path_point.getAttribute('name'));
                
                % Skip default path points
                if startsWith(path_point_name, 'default*','IgnoreCase',true)
                    continue;
                end
                
                socket_elements = path_point.getElementsByTagName('socket_parent_frame');
                
                if socket_elements.getLength() > 0
                    socket_frame = char(socket_elements.item(0).getTextContent());
                    
                    if contains(socket_frame, '/bodyset/rib') && (endsWith(socket_frame, '_L') || endsWith(socket_frame, '_R'))
                        
                        location_elements = path_point.getElementsByTagName('location');
                        
                        if location_elements.getLength() > 0
                            location_str = char(location_elements.item(0).getTextContent());
                            coords = str2num(location_str); 
                            
                            if length(coords) == 3
                                rib_count = rib_count + 1;
                                
                                rib_parts = split(socket_frame, '/');
                                rib_name = rib_parts{end};
                                
                                % Create struct for this path point
                                current_point.path_point_name = path_point_name;
                                current_point.muscle_name = muscle_name;
                                current_point.rib_name = rib_name;
                                current_point.socket_frame = socket_frame;
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
                                    path_point_name, muscle_name, rib_name);
                            end
                        end
                    end
                end
            end
        end
    end
    
    fprintf('Total muscle rib path points found: %d\n', rib_count);
end

