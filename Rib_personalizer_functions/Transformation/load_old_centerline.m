function old_centerline = load_old_centerline(obj_file)
    % Load old centerline from OBJ file format
    
    if ~exist(obj_file, 'file')
        error('Old centerline file not found: %s', obj_file);
    end
    
    vertices = [];
    
    % Read the OBJ file
    fid = fopen(obj_file, 'r');
    if fid == -1
        error('Could not open file: %s', obj_file);
    end
    
    try
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line) && ~isempty(line)
                % Parse vertex lines (start with 'v ')
                if startsWith(strtrim(line), 'v ')
                    coords = sscanf(line, 'v %f %f %f');
                    if length(coords) == 3
                        vertices = [vertices; coords'];
                    end
                end
                % We ignore the line connectivity info for now
            end
        end
    catch ME
        fclose(fid);
        rethrow(ME);
    end
    
    fclose(fid);
    
    if isempty(vertices)
        error('No vertices found in OBJ file: %s', obj_file);
    end
    
    % Create centerline structure
    old_centerline.vertices = vertices;
    old_centerline.num_points = size(vertices, 1);
    
    fprintf('Loaded old centerline: %d points\n', old_centerline.num_points);
end