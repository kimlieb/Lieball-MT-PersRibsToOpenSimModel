function [closest_idx, t_local] = find_closest_centerline_point_CLApproach(point, centerline)
    % Find closest point on centerline and local parameter

    n_points = size(centerline, 1);
    min_dist = inf;
    closest_idx = 1;
    t_local = 0;

    % Check each segment of the centerline
    for i = 1:(n_points-1)
        p1 = centerline(i, :);
        p2 = centerline(i+1, :);

        % Find closest point on this segment
        [seg_point, t_seg] = closest_point_on_segment(point, p1, p2);
        dist = norm(point - seg_point);

        if dist < min_dist
            min_dist = dist;
            closest_idx = i;
            t_local = t_seg;
        end
    end
end
