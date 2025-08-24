function success = update_pathpoint_in_xml(xml_doc, muscle_name, path_point_name, new_location, debug_mode)
    % Update a specific path point location in the XML document
    
    success = false;
    
    try
        % Find all Millard2012EquilibriumMuscle elements
        muscle_elements = xml_doc.getElementsByTagName('Millard2012EquilibriumMuscle');
        
        for muscle_idx = 0:muscle_elements.getLength()-1
            muscle_element = muscle_elements.item(muscle_idx);
            current_muscle_name = char(muscle_element.getAttribute('name'));
            
            % Check if this is the muscle we're looking for
            if strcmp(current_muscle_name, muscle_name)
                if debug_mode
                    fprintf('    Found muscle: %s\n', muscle_name);
                end
                
                % Find all PathPoint elements within this muscle
                path_points = muscle_element.getElementsByTagName('PathPoint');
                
                for pp_idx = 0:path_points.getLength()-1
                    path_point = path_points.item(pp_idx);
                    current_pp_name = char(path_point.getAttribute('name'));
                    
                    % Check if this is the path point we're looking for
                    if strcmp(current_pp_name, path_point_name)
                        if debug_mode
                            fprintf('    Found path point: %s\n', path_point_name);
                        end
                        
                        % Find the location element
                        location_elements = path_point.getElementsByTagName('location');
                        
                        if location_elements.getLength() > 0
                            location_element = location_elements.item(0);
                            
                            % Format new location as string
                            new_location_str = sprintf('%.6f %.6f %.6f', ...
                                new_location(1), new_location(2), new_location(3));
                            
                            % Update the text content
                            location_element.setTextContent(new_location_str);
                            
                            if debug_mode
                                fprintf('    Updated location to: %s\n', new_location_str);
                            end
                            
                            success = true;
                            return;
                        else
                            fprintf('    Warning: No location element found for %s\n', path_point_name);
                        end
                    end
                end
                
                % If we found the muscle but not the path point
                fprintf('    Warning: Path point %s not found in muscle %s\n', path_point_name, muscle_name);
                return;
            end
        end
        
        % If we didn't find the muscle
        fprintf('    Warning: Muscle %s not found in XML\n', muscle_name);
        
    catch ME
        fprintf('    Error updating XML for %s/%s: %s\n', muscle_name, path_point_name, ME.message);
    end
end