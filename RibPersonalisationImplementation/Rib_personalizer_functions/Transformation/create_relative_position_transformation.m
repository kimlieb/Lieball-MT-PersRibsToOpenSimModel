% Modified transformation data creation function
function transform_data = create_relative_position_transformation(old_centerline, new_centerline)
    % Create transformation data for relative position method
    
    fprintf('   Computing relative position transformation...\n');
    
    old_vertices = old_centerline.vertices;
    new_vertices = new_centerline.vertices;
    
    % Check and fix direction if needed
    [old_vertices, new_vertices] = check_and_fix_direction_simple(old_vertices, new_vertices);
    
    % Store transformation data
    transform_data.type = 'relative_position';
    transform_data.old_centerline = old_vertices;
    transform_data.new_centerline = new_vertices;
    transform_data.debug = false; % Set to true for debugging output
    
    % Calculate some statistics
    old_length = compute_centerline_length(old_vertices);
    new_length = compute_centerline_length(new_vertices);
    
    fprintf('   Old centerline length: %.3f mm\n', old_length*1000);
    fprintf('   New centerline length: %.3f mm\n', new_length*1000);
    fprintf('   Length ratio: %.3f\n', new_length/old_length);
    fprintf('   Relative position transformation computed successfully\n');
end