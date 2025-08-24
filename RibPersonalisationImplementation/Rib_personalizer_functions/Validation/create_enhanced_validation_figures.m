function create_enhanced_validation_figures(old_centerlines, transformed_centerlines, personal_centerlines, ...
                                            rib_path_points, transformed_path_points, outputFolder)
    % Create enhanced validation figures with muscle path points for each rib pair
    
    rib_numbers = 1:12;
    sides = {'L', 'R'};
    
    for rib_num = rib_numbers
        fprintf('   Creating enhanced figure for Rib %d...\n', rib_num);
        
        % Create figure for this rib pair
        fig = figure('Position', [100, 100, 1600, 700]);
        
        % Left rib subplot
        subplot(1, 2, 1);
        rib_key_L = sprintf('rib%d_l', rib_num);
        plot_enhanced_rib_comparison(old_centerlines, transformed_centerlines, personal_centerlines, ...
                                     transformed_path_points, rib_key_L, sprintf('Rib %d Left', rib_num));
        
        % Right rib subplot
        subplot(1, 2, 2);
        rib_key_R = sprintf('rib%d_r', rib_num);
        plot_enhanced_rib_comparison(old_centerlines, transformed_centerlines, personal_centerlines, ...
                                     transformed_path_points, rib_key_R, sprintf('Rib %d Right', rib_num));
        
        % Save figure
        sgtitle(sprintf('Rib %d: Centerlines + Muscle Path Points Transformation', rib_num), ...
                'FontSize', 16, 'FontWeight', 'bold');
        
        % Save as both PNG and FIG
        output_name = sprintf('Rib%02d_enhanced_validation', rib_num);
        saveas(fig, fullfile(outputFolder, [output_name '.png']));
        saveas(fig, fullfile(outputFolder, [output_name '.fig']));
        
        % Close figure to save memory
        close(fig);
    end
end