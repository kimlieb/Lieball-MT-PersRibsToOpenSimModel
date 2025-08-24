%% COMPLETE RIB TRANSFORMATION PIPELINE
clc; clear; close all;


% Path to the transformation functions (first code)
addpath('C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_04_ModelAdapterForPersRibs\Rib_personalizer_functions');
addpath('C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_04_ModelAdapterForPersRibs\Rib_personalizer_functions\Transformation');
addpath('C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_04_ModelAdapterForPersRibs\Rib_personalizer_functions\Validation');


% Main script to transform muscle path points from old to new rib coordinate frames

% UPDATE THESE PATHS TO YOUR ACTUAL FILES
osim_input = 'C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_05_OpenSim_ModelAdapted\SGM02\model\SGM02.osim';

oldCenterlinesFolder = 'C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_01_Rib_Generators\08_01_D_CenterlineFromGeneric\genericRibs_CenterlinesFromVTP\';  % Folder with centerline_Rib1L.obj files of generic Ribs
newCenterlinesTxt = 'C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_01_Rib_Generators\08_01_C_OpenSim_RibGenerators\translated_centerlines_finalNewSegmentation.txt';  % Your txt file with new centerlines

% Choose how to name your output Files
osim_output_genericRibs = 'C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_05_OpenSim_ModelAdapted\SGM02\model\SGM02_GenericRibsAdaptedMusclePP_relativePosition.osim';
osim_output_AdaptedRibsAndMusclePPs = "C:\Users\kimli\OneDrive - Universitaet Bern\MasterThesis_Kim\08_MT_final\08_05_OpenSim_ModelAdapted\SGM02\model\SGM02_personalisedRibsAdaptedMusclePP_relativePosition.osim"; 

% Choose which transformation method to use:
transformation_method = 'relative_position'; % 'relative_position' OR 'tps', tps still needs work

% choose if you want validation figures of the rib transformation and the
% muscle pathpoints
validation_option = 'no'; % yes or no
% Output folder for visualization figures
outputFolder = 'rib_transformation_validation_figures/';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


transform_rib_pathpoints(osim_input, osim_output_genericRibs, newCenterlinesTxt, oldCenterlinesFolder, osim_output_AdaptedRibsAndMusclePPs, transformation_method, validation_option, outputFolder);

%% MAIN TRANSFORMATION FUNCTION
function transform_rib_pathpoints(osim_filename, osim_output_genericRibs, newCenterlinesTxt, oldCenterlinesFolder, osim_output_AdaptedRibsAndMusclePPs, transformation_method, validation_option, validation_OutputFolder)
    
    % Step 1: Extract rib path points
    fprintf('Step 1: Extracting rib path points...\n');
    rib_data = extract_rib_pathpoints(osim_filename);
    
    if isempty(rib_data)
        error('No rib path points found in the OSIM file.');
    end
    

    % Step 2: Compute transformations (method depends on selection)
    fprintf('Step 2: Computing rib transformations...\n');
    switch transformation_method
        case 'relative_position'
            transformations = compute_rib_transformations_relativePosition(newCenterlinesTxt, oldCenterlinesFolder);           
        case 'tps'
            transformations = compute_rib_transformations_usingTPS(newCenterlinesTxt, oldCenterlinesFolder);
        otherwise
            error('Unknown transformation method: %s', transformation_method);
    end
    
    % Step 3: Transform path points and update the osim File
    fprintf('Step 3: Computing new path point locations...\n');

    switch transformation_method
        case 'relative_position'
            updated_locations = transform_pathpoints_relativePosition(rib_data, transformations, ...
                'visualize', false, 'debug', true, ...
                'osim_input', osim_filename, 'osim_output', osim_output_genericRibs);
            
        case 'tps'
            updated_locations = transform_pathpoints_usingTPS(rib_data, transformations, ...
                'visualize', false, 'debug', false, ...
                'osim_input', osim_filename, 'osim_output', osim_output_genericRibs);            
        otherwise
            error('Unknown transformation method: %s', transformation_method);
    end

    % Step 4: Replace the rib meshes
    modify_rib_meshes_underscore(osim_output_genericRibs, osim_output_AdaptedRibsAndMusclePPs);
    
    fprintf('=== TRANSFORMATION PIPELINE COMPLETE ===\n');
    fprintf('Transformation complete! Updated file with personalised ribs saved as: %s\n', osim_output_AdaptedRibsAndMusclePPs);

    % Step 5 (optional): Validation of the transformation

    switch validation_option
        case 'yes'
            
            % Step 5a: Convert array format to struct format for compatibility with visualization
            transformed_path_points = convert_array_to_struct_format(updated_locations);
            
            % Step 5b: Load centerlines
            [old_centerlines, transformed_centerlines, personal_centerlines] = transform_all_centerlines(transformations, newCenterlinesTxt, oldCenterlinesFolder);
            
            % Step 5c: Create enhanced validation visualizations
            fprintf('Step 5: Creating enhanced validation visualizations...\n');
            create_enhanced_validation_figures(old_centerlines, transformed_centerlines, personal_centerlines, ...
                                               rib_data, transformed_path_points, validation_OutputFolder);
            
            fprintf('=== ENHANCED VALIDATION COMPLETE ===\n');
            fprintf('Figures saved in: %s\n', validation_OutputFolder);
            
        case 'no'
            fprintf('DONE');            
        otherwise
            fprintf('Please choose validation option yes or no \n');
    end
    

end

