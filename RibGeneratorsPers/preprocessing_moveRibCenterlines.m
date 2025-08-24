function translate_rib_centerlines_to_origin(filename, filename_translatedData)
    % Read the txt file
    data = readtable(filename);
    
    % Create a copy for the translated data
    translated_data = data;
    
    % Get unique combinations of rib_index and side
    unique_ribs = unique(data(:, {'rib_index', 'side'}));
    
    % Process each rib (left and right sides separately)
    for i = 1:height(unique_ribs)
        rib_idx = unique_ribs.rib_index(i);
        side = unique_ribs.side(i);
        
        % Find all points for this specific rib and side
        mask = (data.rib_index == rib_idx) & strcmp(data.side, side);
        
        % Extract the points for this rib/side
        rib_points = data(mask, :);
        
        % Sort by point_index to ensure correct order
        rib_points = sortrows(rib_points, 'point_index');
        
        % Get the first point (point_index = 0) coordinates
        first_point = rib_points(1, :);
        first_coords = [first_point.x, first_point.y, first_point.z];
        
        % Translate all points by subtracting the first point
        for j = 1:height(rib_points)
            % Find the corresponding row in the translated_data
            row_mask = (translated_data.rib_index == rib_idx) & ...
                      strcmp(translated_data.side, side) & ...
                      (translated_data.point_index == rib_points.point_index(j));
            
            % Apply translation
            translated_data.x(row_mask) = rib_points.x(j) - first_coords(1);
            translated_data.y(row_mask) = rib_points.y(j) - first_coords(2);
            translated_data.z(row_mask) = rib_points.z(j) - first_coords(3);
        end
        
        % Display progress
        fprintf('Translated Rib %d%s to origin\n', rib_idx, side{1});
    end
    
    % Save the translated data
    writetable(translated_data, filename_translatedData);
    
    % Display summary
    fprintf('\nTranslation complete!\n');
    fprintf('Original file: %s\n', filename);
    fprintf('Translated file: %s\n', filename_translatedData);
    fprintf('Total ribs processed: %d\n', height(unique_ribs));
    
    % Optional: Create visualization
    create_visualization(data, translated_data);
end

function create_visualization(original_data, translated_data)
    % Create visualization comparing original and translated data
    figure('Position', [100, 100, 1200, 600]);
    
    % Plot original data
    subplot(1, 2, 1);
    hold on;
    grid on;
    
    % Plot first few ribs for clarity
    ribs_to_plot = unique(original_data.rib_index);
    colors = ['r', 'b', 'g', 'c', 'm', 'y'];
    
    for i = 1:min(3, length(ribs_to_plot))  % Show first 3 ribs
        rib_idx = ribs_to_plot(i);
        color_idx = mod(i-1, length(colors)) + 1;
        
        % Plot left side
        mask_L = (original_data.rib_index == rib_idx) & strcmp(original_data.side, 'L');
        if any(mask_L)
            rib_data_L = original_data(mask_L, :);
            rib_data_L = sortrows(rib_data_L, 'point_index');
            plot3(rib_data_L.x, rib_data_L.y, rib_data_L.z, ...
                  [colors(color_idx), 'o-'], 'LineWidth', 2, ...
                  'DisplayName', sprintf('Rib %d L', rib_idx));
        end
        
        % Plot right side
        mask_R = (original_data.rib_index == rib_idx) & strcmp(original_data.side, 'R');
        if any(mask_R)
            rib_data_R = original_data(mask_R, :);
            rib_data_R = sortrows(rib_data_R, 'point_index');
            plot3(rib_data_R.x, rib_data_R.y, rib_data_R.z, ...
                  [colors(color_idx), 's--'], 'LineWidth', 2, ...
                  'DisplayName', sprintf('Rib %d R', rib_idx));
        end
    end
    
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Original Centerlines');
    legend('Location', 'best');
    axis equal;
    
    % Plot translated data
    subplot(1, 2, 2);
    hold on;
    grid on;
    
    for i = 1:min(3, length(ribs_to_plot))  % Show first 3 ribs
        rib_idx = ribs_to_plot(i);
        color_idx = mod(i-1, length(colors)) + 1;
        
        % Plot left side
        mask_L = (translated_data.rib_index == rib_idx) & strcmp(translated_data.side, 'L');
        if any(mask_L)
            rib_data_L = translated_data(mask_L, :);
            rib_data_L = sortrows(rib_data_L, 'point_index');
            plot3(rib_data_L.x, rib_data_L.y, rib_data_L.z, ...
                  [colors(color_idx), 'o-'], 'LineWidth', 2, ...
                  'DisplayName', sprintf('Rib %d L', rib_idx));
        end
        
        % Plot right side
        mask_R = (translated_data.rib_index == rib_idx) & strcmp(translated_data.side, 'R');
        if any(mask_R)
            rib_data_R = translated_data(mask_R, :);
            rib_data_R = sortrows(rib_data_R, 'point_index');
            plot3(rib_data_R.x, rib_data_R.y, rib_data_R.z, ...
                  [colors(color_idx), 's--'], 'LineWidth', 2, ...
                  'DisplayName', sprintf('Rib %d R', rib_idx));
        end
    end
    
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Translated Centerlines (Starting at Origin)');
    legend('Location', 'best');
    axis equal;
    
    % Add a marker at origin for reference
    plot3(0, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', ...
          'DisplayName', 'Origin');
    
    sgtitle('Rib Centerline Translation Comparison');
end

function verify_translation(filename)
    % Verification function to check if translation worked correctly
    data = readtable(filename);
    
    % Check that all first points (point_index = 0) are at origin
    first_points = data(data.point_index == 0, :);
    
    fprintf('Verification Results:\n');
    fprintf('==================\n');
    
    for i = 1:height(first_points)
        rib_idx = first_points.rib_index(i);
        side = first_points.side(i);
        coords = [first_points.x(i), first_points.y(i), first_points.z(i)];
        
        distance_from_origin = norm(coords);
        
        fprintf('Rib %d%s: (%.6f, %.6f, %.6f) - Distance from origin: %.6f\n', ...
                rib_idx, side{1}, coords(1), coords(2), coords(3), distance_from_origin);
    end
    
    % Check if all starting points are at origin (within numerical precision)
    tolerance = 1e-10;
    all_at_origin = all(vecnorm([first_points.x, first_points.y, first_points.z], 2, 2) < tolerance);
    
    if all_at_origin
        fprintf('\n✓ SUCCESS: All centerlines start at origin!\n');
    else
        fprintf('\n✗ ERROR: Some centerlines do not start at origin.\n');
    end
end

%% Usage
filename_centerlines = 'Input_CenterlineFile\rib_centerlines_finalNewSegmentation.txt';
filename_translatedData = 'translated_centerlines_finalNewSegmentation.txt'; 

translate_rib_centerlines_to_origin(filename_centerlines, filename_translatedData);
verify_translation(filename_translatedData);