function resampled = resample_centerline(vertices, target_points)
    % Resample centerline to have target number of points
    
    num_points = size(vertices, 1);
    
    if num_points == target_points
        resampled = vertices;
        return;
    end
    
    % Compute cumulative arc length
    distances = sqrt(sum(diff(vertices).^2, 2));
    cumulative_length = [0; cumsum(distances)];
    total_length = cumulative_length(end);
    
    % Create new parameter values
    new_params = linspace(0, total_length, target_points)';
    
    % Interpolate each coordinate
    resampled = zeros(target_points, 3);
    for dim = 1:3
        resampled(:, dim) = interp1(cumulative_length, vertices(:, dim), new_params, 'linear');
    end
end