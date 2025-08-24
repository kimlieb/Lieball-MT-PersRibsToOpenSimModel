function rib_key = parse_rib_key(rib_name)
    % Parse rib name and return standardized key format
    % Handles various formats: rib5_R, rib5_r, Rib5_R, etc.
    
    rib_key = '';
    rib_name_lower = lower(rib_name);
    
    % Try different parsing patterns
    patterns = {
        'rib(\d+)_([lr])',      % rib5_r, rib5_l
        'rib(\d+)([lr])',       % rib5r, rib5l  
        '(\d+)_rib_([lr])',     % 5_rib_r, 5_rib_l
        'costa(\d+)_([lr])'     % costa5_r, costa5_l
    };
    
    for i = 1:length(patterns)
        match = regexp(rib_name_lower, patterns{i}, 'tokens');
        if ~isempty(match)
            rib_num = str2double(match{1}{1});
            side = match{1}{2};
            rib_key = sprintf('rib%d_%s', rib_num, side);
            return;
        end
    end
    
    % If no pattern matches, try to extract at least the number
    num_match = regexp(rib_name_lower, '(\d+)', 'tokens');
    if ~isempty(num_match)
        rib_num = str2double(num_match{1}{1});
        
        % Guess side from name
        if contains(rib_name_lower, 'r')
            side = 'r';
        elseif contains(rib_name_lower, 'l')
            side = 'l';
        else
            fprintf('Warning: Could not determine side for %s\n', rib_name);
            return;
        end
        
        rib_key = sprintf('rib%d_%s', rib_num, side);
    end
end