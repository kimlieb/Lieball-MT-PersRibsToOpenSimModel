function point = interpolate_on_centerline(t, centerline)
    % Interpolate point at parameter t along centerline

    n_points = size(centerline, 1);

    if t <= 0
        point = centerline(1, :);
        return;
    elseif t >= 1
        point = centerline(end, :);
        return;
    end

    % Calculate cumulative arc lengths
    distances = sqrt(sum(diff(centerline).^2, 2));
    cumulative = [0; cumsum(distances)];
    total_length = cumulative(end);

    % Find target arc length
    target_length = t * total_length;

    % Find which segment contains this arc length
    for i = 1:(n_points-1)
        if cumulative(i+1) >= target_length
            % Interpolate within this segment
            segment_start = cumulative(i);
            segment_length = cumulative(i+1) - segment_start;

            if segment_length > 1e-10
                local_t = (target_length - segment_start) / segment_length;
            else
                local_t = 0;
            end

            point = centerline(i, :) + local_t * (centerline(i+1, :) - centerline(i, :));
            return;
        end
    end

    % Fallback
    point = centerline(end, :);
end

