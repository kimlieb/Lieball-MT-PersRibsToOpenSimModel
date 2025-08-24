
function total_length = compute_centerline_length(vertices)
    % Compute total arc length of centerline
    if size(vertices, 1) < 2
        total_length = 0;
        return;
    end
    distances = sqrt(sum(diff(vertices).^2, 2));
    total_length = sum(distances);
end