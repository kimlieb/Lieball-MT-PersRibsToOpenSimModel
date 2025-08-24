function corresponding_point = find_corresponding_centerline_point(relative_position, centerline)
    % Find point on centerline at given relative position
    
    if size(centerline, 1) < 2
        corresponding_point = centerline(1, :);
        return;
    end
    
    % Calculate cumulative arc lengths
    segment_lengths = sqrt(sum(diff(centerline).^2, 2));
    cumulative_lengths = [0; cumsum(segment_lengths)];
    total_length = cumulative_lengths(end);
    
    if total_length == 0
        corresponding_point = centerline(1, :);
        return;
    end
    
    % Find target arc length
    target_length = relative_position * total_length;
    
    % Find which segment contains this arc length
    segment_idx = find(cumulative_lengths <= target_length, 1, 'last');
    
    if segment_idx >= size(centerline, 1)
        corresponding_point = centerline(end, :);
        return;
    end
    
    % Interpolate within the segment
    if segment_idx == length(cumulative_lengths)
        corresponding_point = centerline(end, :);
    else
        % Linear interpolation between segment_idx and segment_idx+1
        length_in_segment = target_length - cumulative_lengths(segment_idx);
        segment_length = segment_lengths(segment_idx);
        
        if segment_length > 0
            t = length_in_segment / segment_length;
            corresponding_point = (1-t) * centerline(segment_idx, :) + t * centerline(segment_idx+1, :);
        else
            corresponding_point = centerline(segment_idx, :);
        end
    end
end