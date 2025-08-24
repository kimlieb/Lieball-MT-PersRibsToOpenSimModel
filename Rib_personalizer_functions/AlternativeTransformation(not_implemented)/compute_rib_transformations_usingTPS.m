function transformations = compute_rib_transformations_usingTPS(newCenterlinetxt, oldCenterlinesFolder)
    % Compute transformations for all rib pairs using centerlines
    
    transformations = struct();
    
    % Read the new centerlines from the txt file
    fprintf('Reading new centerlines from: %s\n', newCenterlinetxt);
    new_centerlines = load_new_centerlines(newCenterlinetxt);
    
    % Define rib names (12 ribs, left and right)
    rib_numbers = 1:12;
    sides = {'L', 'R'};
    
    % Process each rib pair
    for rib_num = rib_numbers
        for side_idx = 1:length(sides)
            side = sides{side_idx};
            
            % Construct filenames
            old_rib_filename = sprintf('centerline_Rib%d%s.obj', rib_num, side);
            old_file_path = fullfile(oldCenterlinesFolder, old_rib_filename);
            
            try
                % Load old centerline
                fprintf('Loading old centerline for Rib%d%s...\n', rib_num, side);
                old_centerline = load_old_centerline(old_file_path);
                
                % Get corresponding new centerline
                rib_key = sprintf('rib%d_%s', rib_num, lower(side));
                if isfield(new_centerlines, rib_key)
                    new_centerline = new_centerlines.(rib_key);
                    
                    % Compute transformation using linear blend method
                    fprintf('Computing linear blend transformation for %s...\n', rib_key);
                    % transform_data = create_centerline_transformation_linear_blend(old_centerline, new_centerline);
                    transform_data = create_centerline_transformation(old_centerline, new_centerline); % KIM
                    
                    % Store transformation data
                    transformations.(rib_key) = transform_data;
                    
                    fprintf('Successfully computed transformation for %s\n', rib_key);
                else
                    fprintf('Warning: No new centerline found for %s\n', rib_key);
                end
                
            catch ME
                fprintf('Error processing Rib%d%s: %s\n', rib_num, side, ME.message);
            end
        end
    end
    
    fprintf('Total transformations computed: %d\n', length(fieldnames(transformations)));
end










