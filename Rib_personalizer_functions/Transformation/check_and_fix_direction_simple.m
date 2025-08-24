function [old_vertices, new_vertices] = check_and_fix_direction_simple(old_vertices, new_vertices)
    % Simple direction check and fix
    
    % Calculate total distance for current orientation
    dist_normal = norm(old_vertices(1,:) - new_vertices(1,:)) + ...
                  norm(old_vertices(end,:) - new_vertices(end,:));
    
    % Calculate total distance if we reverse new centerline
    dist_reversed = norm(old_vertices(1,:) - new_vertices(end,:)) + ...
                    norm(old_vertices(end,:) - new_vertices(1,:));
    
    % If reversed orientation gives better correspondence, reverse it
    if dist_reversed < dist_normal
        new_vertices = flipud(new_vertices);
        fprintf('   Reversed new centerline for better correspondence\n');
    end
end