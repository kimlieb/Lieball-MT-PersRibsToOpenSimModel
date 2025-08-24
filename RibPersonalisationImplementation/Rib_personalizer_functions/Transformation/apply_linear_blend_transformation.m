function P_new = apply_linear_blend_transformation(P_old, transform_data)
    % Apply linear blend transformation along centerline

    old_centerline = transform_data.old_centerline;
    new_centerline = transform_data.new_centerline;

    % Find closest point on old centerline and get parameter t
    [closest_idx, t_local] = find_closest_centerline_point(P_old, old_centerline);

    % Calculate global parameter t along entire centerline [0,1]
    t_global = calculate_global_parameter(closest_idx, t_local, old_centerline);

    % Find corresponding point on new centerline
    new_centerline_point = interpolate_on_centerline(t_global, new_centerline);

    % Calculate offset from old centerline
    old_centerline_point = interpolate_on_centerline(t_global, old_centerline);
    offset_vector = P_old - old_centerline_point;

    % Apply scaled offset to new centerline point
    scale_factor = transform_data.new_length / transform_data.old_length;
    P_new = new_centerline_point + offset_vector * scale_factor;
end