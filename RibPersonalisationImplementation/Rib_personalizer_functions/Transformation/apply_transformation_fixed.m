

% Modified apply_transformation_fixed function to handle different transformation types
function P_new = apply_transformation_fixed(P_old, transform_data)
    % Apply the appropriate transformation to a point
    
    switch transform_data.type
        case 'relative_position'
            P_new = apply_transformation_relative_position(P_old, transform_data);
        case 'tps'
            P_new = apply_tps_transformation(P_old, transform_data);
        case 'simple'
            P_new = apply_simple_transformation(P_old, transform_data);
        case 'linear_blend'
            P_new = apply_linear_blend_transformation(P_old, transform_data);
        otherwise
            error('Unknown transformation type: %s', transform_data.type);
    end
end