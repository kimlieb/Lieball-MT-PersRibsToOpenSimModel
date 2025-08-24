function P_new = apply_simple_transformation(P_old, transform_data)
    % Apply simple transformation to a point

    % Center the point
    P_centered = P_old - transform_data.old_centroid;

    % Apply scaling and translation
    P_new = P_centered .* transform_data.scale_factors + transform_data.new_centroid;
end
