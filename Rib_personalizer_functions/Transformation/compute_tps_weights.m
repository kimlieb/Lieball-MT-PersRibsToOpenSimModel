function [weights, affine] = compute_tps_weights(source_points, target_points)
    % Compute Thin Plate Spline weights
    
    n = size(source_points, 1);
    
    % Compute the radial basis function matrix
    K = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if i ~= j
                r_sq = sum((source_points(i,:) - source_points(j,:)).^2);
                if r_sq > 0
                    K(i,j) = r_sq * log(sqrt(r_sq));
                end
            end
        end
    end
    
    % Build the system matrix
    P = [ones(n,1), source_points];  % [1, x, y, z]
    L = [K, P; P', zeros(4,4)];
    
    % Build right-hand side for each coordinate
    weights = zeros(n+4, 3);
    for dim = 1:3
        rhs = [target_points(:,dim); zeros(4,1)];
        
        % Solve the linear system with regularization
        try
            w = L \ rhs;
        catch
            % Add small regularization if matrix is singular
            reg_L = L + 1e-10 * eye(size(L));
            w = reg_L \ rhs;
        end
        
        weights(:, dim) = w;
    end
    
    % Separate weights and affine parameters
    affine = weights(end-3:end, :);
    weights = weights(1:end-4, :);
end