function plot_enhanced_rib_comparison(old_centerlines, transformed_centerlines, personal_centerlines, ...
                                      transformed_path_points, rib_key, title_str)
    % Plot enhanced comparison including centerlines and muscle path points
    
    hold on;
    
    % Plot old centerline (blue)
    if isfield(old_centerlines, rib_key)
        old_points = old_centerlines.(rib_key);
        plot3(old_points(:,1), old_points(:,2), old_points(:,3), 'b-', 'LineWidth', 2, 'DisplayName', 'Original Centerline');
        scatter3(old_points(1,1), old_points(1,2), old_points(1,3), 80, 'b', 'filled', 'DisplayName', 'Original Start');
        scatter3(old_points(end,1), old_points(end,2), old_points(end,3), 80, 'b', '^', 'filled', 'DisplayName', 'Original End');
    end
    
    % Plot transformed centerline (red)
    if isfield(transformed_centerlines, rib_key)
        trans_points = transformed_centerlines.(rib_key);
        plot3(trans_points(:,1), trans_points(:,2), trans_points(:,3), 'r--', 'LineWidth', 2, 'DisplayName', 'Transformed Centerline');
        scatter3(trans_points(1,1), trans_points(1,2), trans_points(1,3), 80, 'r', 'filled', 'DisplayName', 'Transformed Start');
        scatter3(trans_points(end,1), trans_points(end,2), trans_points(end,3), 80, 'r', '^', 'filled', 'DisplayName', 'Transformed End');
    end
    
    % Plot personal centerline (green)
    if isfield(personal_centerlines, rib_key)
        pers_points = personal_centerlines.(rib_key);
        plot3(pers_points(:,1), pers_points(:,2), pers_points(:,3), 'g:', 'LineWidth', 3, 'DisplayName', 'Personal/Target Centerline');
        scatter3(pers_points(1,1), pers_points(1,2), pers_points(1,3), 80, 'g', 'filled', 'DisplayName', 'Personal Start');
        scatter3(pers_points(end,1), pers_points(end,2), pers_points(end,3), 80, 'g', '^', 'filled', 'DisplayName', 'Personal End');
    end
    
    % Plot muscle path points - ORIGINAL (cyan circles)
    if isfield(transformed_path_points, rib_key) && ~isempty(transformed_path_points.(rib_key))
        path_points_data = transformed_path_points.(rib_key);
        
        % Extract original muscle path points
        original_points = [path_points_data.original_location];
        original_points = reshape(original_points, 3, [])'; % Convert to Nx3 matrix
        
        if ~isempty(original_points)
            scatter3(original_points(:,1), original_points(:,2), original_points(:,3), ...
                    120, 'c', 'o', 'LineWidth', 2, 'DisplayName', 'Original Muscle Points');
            
            % Add muscle names as text labels for original points
            for i = 1:size(original_points, 1)
                text(original_points(i,1), original_points(i,2), original_points(i,3), ...
                     sprintf(' %s', path_points_data(i).muscle_name), ...
                     'FontSize', 8, 'Color', 'c', 'FontWeight', 'bold');
            end
        end
    end
    
    % Plot muscle path points - TRANSFORMED (magenta squares)
    if isfield(transformed_path_points, rib_key) && ~isempty(transformed_path_points.(rib_key))
        path_points_data = transformed_path_points.(rib_key);
        
        % Extract transformed muscle path points
        transformed_points = [path_points_data.transformed_location];
        transformed_points = reshape(transformed_points, 3, [])'; % Convert to Nx3 matrix
        
        if ~isempty(transformed_points)
            scatter3(transformed_points(:,1), transformed_points(:,2), transformed_points(:,3), ...
                    120, 'm', 's', 'filled', 'DisplayName', 'Transformed Muscle Points');
            
            % Add muscle names as text labels for transformed points
            for i = 1:size(transformed_points, 1)
                text(transformed_points(i,1), transformed_points(i,2), transformed_points(i,3), ...
                     sprintf(' %s', path_points_data(i).muscle_name), ...
                     'FontSize', 8, 'Color', 'm', 'FontWeight', 'bold');
            end
        end
        
        % Draw arrows connecting original to transformed muscle points
        if exist('original_points', 'var') && ~isempty(original_points) && ~isempty(transformed_points)
            for i = 1:min(size(original_points, 1), size(transformed_points, 1))
                % Draw arrow from original to transformed
                arrow_start = original_points(i, :);
                arrow_end = transformed_points(i, :);
                
                % Plot arrow line
                plot3([arrow_start(1), arrow_end(1)], ...
                      [arrow_start(2), arrow_end(2)], ...
                      [arrow_start(3), arrow_end(3)], ...
                      'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
                
                % Add arrowhead (simplified)
                direction = arrow_end - arrow_start;
                if norm(direction) > 1e-6
                    direction = direction / norm(direction);
                    arrowhead_start = arrow_end - 0.002 * direction; % 2mm back from end
                    plot3([arrowhead_start(1), arrow_end(1)], ...
                          [arrowhead_start(2), arrow_end(2)], ...
                          [arrowhead_start(3), arrow_end(3)], ...
                          'k-', 'LineWidth', 2, 'HandleVisibility', 'off');
                end
            end
        end
    end
    
    % Formatting
    title(title_str, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    grid on;
    axis equal;
    
    % Improve viewing angle
    view(45, 30);
    
    % Add legend (limit entries to avoid clutter)
    legend('Location', 'best', 'FontSize', 9);
    
    % Add muscle path points statistics if available
    if isfield(transformed_path_points, rib_key) && ~isempty(transformed_path_points.(rib_key))
        add_muscle_statistics_textbox(transformed_path_points.(rib_key));
    end
    
    hold off;
end

function add_muscle_statistics_textbox(path_points_data)
    % Add a text box with muscle path point transformation statistics
    
    if isempty(path_points_data)
        return;
    end
    
    % Calculate transformation distances for muscle points
    distances = [];
    for i = 1:length(path_points_data)
        original = path_points_data(i).original_location;
        transformed = path_points_data(i).transformed_location;
        dist = norm(transformed - original);
        distances = [distances; dist];
    end
    
    if ~isempty(distances)
        mean_dist = mean(distances);
        max_dist = max(distances);
        num_muscles = length(path_points_data);
        
        % Create statistics text
        stats_text = sprintf(['Muscle Path Points:\n' ...
                             'Count: %d muscles\n' ...
                             'Mean movement: %.3f mm\n' ...
                             'Max movement: %.3f mm'], ...
                             num_muscles, mean_dist*1000, max_dist*1000);
        
        % Add text box
        text(0.02, 0.02, stats_text, 'Units', 'normalized', ...
             'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
             'BackgroundColor', 'yellow', 'EdgeColor', 'black', ...
             'FontSize', 9, 'FontWeight', 'bold');
    end
end
