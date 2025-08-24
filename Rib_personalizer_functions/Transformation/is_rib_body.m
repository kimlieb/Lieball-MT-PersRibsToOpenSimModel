function is_rib = is_rib_body(body_name)
    % Check if a body name corresponds to a rib
    body_name_lower = lower(body_name);
    rib_patterns = {'rib', 'costa', 'rib_', '_rib', 'ribs'};
    
    is_rib = false;
    for i = 1:length(rib_patterns)
        if contains(body_name_lower, rib_patterns{i})
            is_rib = true;
            break;
        end
    end
    
    % Additional check for numbered ribs
    if ~is_rib
        if ~isempty(regexp(body_name_lower, 'rib\d+', 'once')) || ...
           ~isempty(regexp(body_name_lower, '\d+.*rib', 'once'))
            is_rib = true;
        end
    end
end