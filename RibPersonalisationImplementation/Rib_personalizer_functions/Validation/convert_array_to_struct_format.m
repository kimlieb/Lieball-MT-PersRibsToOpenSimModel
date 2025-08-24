function transformed_path_points = convert_array_to_struct_format(transformed_array)
    % Convert the array format from transform_pathpoints_usingCenterlines 
    % to the struct format expected by the visualization functions
    
    transformed_path_points = struct();
    
    % Initialize structure for each rib
    rib_numbers = 1:12;
    sides = {'L', 'R'};
    for rib_num = rib_numbers
        for side_idx = 1:length(sides)
            side = sides{side_idx};
            rib_key = sprintf('rib%d_%s', rib_num, lower(side));
            transformed_path_points.(rib_key) = [];
        end
    end
    
    % Convert each transformed point
    for i = 1:length(transformed_array)
        point_data = transformed_array(i);
        
        % Parse rib information from rib_name
        rib_name_lower = lower(point_data.rib_name);
        rib_match = regexp(rib_name_lower, 'rib(\d+)_([lr])', 'tokens');
        
        if ~isempty(rib_match)
            rib_num = str2double(rib_match{1}{1});
            side = rib_match{1}{2};
            rib_key = sprintf('rib%d_%s', rib_num, side);
        else
            fprintf('Warning: Could not parse rib name: %s\n', point_data.rib_name);
            continue;
        end
        
        % Create data structure in expected format
        converted_data.muscle_name = point_data.muscle_name;
        converted_data.path_point_name = point_data.path_point_name;
        converted_data.original_location = point_data.old_location;
        converted_data.transformed_location = point_data.new_location;
        
        % Add to appropriate rib structure
        if isempty(transformed_path_points.(rib_key))
            transformed_path_points.(rib_key) = converted_data;
        else
            transformed_path_points.(rib_key)(end+1) = converted_data;
        end
    end
    
    % Print summary
    total_points = 0;
    field_names = fieldnames(transformed_path_points);
    for i = 1:length(field_names)
        rib_key = field_names{i};
        num_points = length(transformed_path_points.(rib_key));
        if num_points > 0
            fprintf('   %s: %d muscle path points\n', rib_key, num_points);
            total_points = total_points + num_points;
        end
    end
    fprintf('   Total converted muscle path points: %d\n', total_points);
end
