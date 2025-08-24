function t_global = calculate_global_parameter(closest_idx, t_local, centerline)
    % Convert local segment parameter to global centerline parameter

    n_points = size(centerline, 1);

    % Calculate cumulative arc lengths
    distances = sqrt(sum(diff(centerline).^2, 2));
    cumulative = [0; cumsum(distances)];
    total_length = cumulative(end);

    if total_length < 1e-10
        t_global = 0;
        return;
    end

    % Arc length up to closest segment
    arc_to_segment = cumulative(closest_idx);

    % Arc length within the segment
    if closest_idx < n_points
        segment_length = distances(closest_idx);
        arc_within_segment = t_local * segment_length;
    else
        arc_within_segment = 0;
    end

    % Global parameter
    t_global = (arc_to_segment + arc_within_segment) / total_length;
    t_global = max(0, min(1, t_global));
end