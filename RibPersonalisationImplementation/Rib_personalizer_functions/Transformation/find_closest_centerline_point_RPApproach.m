


% for relative position approach
function [closest_point, relative_position, closest_idx] = find_closest_centerline_point_RPApproach(P, centerline)
    % Find closest point on centerline and its relative position along the curve
    
    distances = sqrt(sum((centerline - P).^2, 2));
    [~, closest_idx] = min(distances);
    closest_point = centerline(closest_idx, :);
    
    % Calculate relative position along centerline (0 = start, 1 = end)
    % Using arc length parameterization
    if size(centerline, 1) > 1
        segment_lengths = sqrt(sum(diff(centerline).^2, 2));
        cumulative_lengths = [0; cumsum(segment_lengths)];
        total_length = cumulative_lengths(end);
        
        if total_length > 0
            relative_position = cumulative_lengths(closest_idx) / total_length;
        else
            relative_position = 0;
        end
    else
        relative_position = 0;
    end
    
    % Ensure relative_position is in [0, 1]
    relative_position = max(0, min(1, relative_position));
end