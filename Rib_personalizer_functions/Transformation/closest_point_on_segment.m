function [closest_point, t] = closest_point_on_segment(point, p1, p2)
    % Find closest point on line segment p1-p2

    v = p2 - p1;
    w = point - p1;

    if norm(v) < 1e-10
        closest_point = p1;
        t = 0;
        return;
    end

    t = dot(w, v) / dot(v, v);
    t = max(0, min(1, t));  % Clamp to [0,1]

    closest_point = p1 + t * v;
end
