function P_new = apply_tps_transformation(P_old, transform_data)
    % Apply TPS transformation to a point

    source_points = transform_data.source_points;
    weights = transform_data.tps_weights;
    affine = transform_data.tps_affine;

    n = size(source_points, 1);
    P_new = zeros(size(P_old));

    % For each output coordinate
    for dim = 1:3
        % Affine part: a0 + a1*x + a2*y + a3*z
        result = affine(1, dim) + sum(affine(2:4, dim)' .* P_old);

        % Radial basis function part
        for i = 1:n
            r_sq = sum((P_old - source_points(i,:)).^2);
            if r_sq > 0
                rbf_val = r_sq * log(sqrt(r_sq));
            else
                rbf_val = 0;
            end
            result = result + weights(i, dim) * rbf_val;
        end

        P_new(dim) = result;
    end
end
