function P_new = apply_transformation_relative_position(P_old, transform_data)
    % Transform point using relative position approach: P = C + v
    % where P is point to transform, C is closest centerline point, v is offset
    % P_new = C_new + v_old (preserve offset vector)
    
    old_centerline = transform_data.old_centerline;
    new_centerline = transform_data.new_centerline;
    
    % Step 1: Find closest point on old centerline to P_old
    [C_old, relative_position, closest_idx] = find_closest_centerline_point_RPApproach(P_old, old_centerline);
    
    % Step 2: Calculate offset vector v = P_old - C_old
    v_old = P_old - C_old;
    
    % Step 3: Find corresponding point on new centerline at same relative position
    C_new = find_corresponding_centerline_point(relative_position, new_centerline);
    
    % Step 4: Apply transformation: P_new = C_new + v_old
    P_new = C_new + v_old;
    
    % Optional: Add some debugging output
    if transform_data.debug
        fprintf('   P_old: [%.3f, %.3f, %.3f]\n', P_old);
        fprintf('   C_old: [%.3f, %.3f, %.3f] (rel_pos: %.3f)\n', C_old, relative_position);
        fprintf('   v_old: [%.3f, %.3f, %.3f] (offset magnitude: %.3f mm)\n', v_old, norm(v_old)*1000);
        fprintf('   C_new: [%.3f, %.3f, %.3f]\n', C_new);
        fprintf('   P_new: [%.3f, %.3f, %.3f]\n', P_new);
        fprintf('   Movement: %.3f mm\n', norm(P_new - P_old)*1000);
    end
end
