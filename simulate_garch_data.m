function Y = simulate_garch_data(p, T, alpha, beta, mu, Omega, a, b, c)
    % Simulate data for a multivariate GARCH(1,1) model with specified coefficients
    % p - Number of dimensions
    % T - Number of time points
    % alpha, beta - GARCH parameters (vectors of length p)
    % mu - Mean vector (p x 1)
    % Omega - Unconditional covariance matrix (p x p)
    % a, b, c - Coefficients for A, B, and C matrices

    % Initialize variables
    Y = zeros(p, T);
    Sigma_t = Omega; % Initial covariance matrix
    
    % Construct the coefficient matrices
    A_diag = alpha;
    B_diag = beta;
    C_diag = ones(p, 1); % Diagonal elements for C are ones

    A = Coeff_mat_a2b2(a, A_diag);
    B = Coeff_mat_a2b2(b, B_diag);
    C = Coeff_mat_a2b2(c, C_diag); % Using c for C

    % Generate the data
    for t = 2:T
        % Generate multivariate normal data with the current covariance matrix
        Y(:, t) = mvnrnd(mu, Sigma_t)';

        % Update the covariance matrix for the next time point
        Sigma_t = (ones(p) - A - B) .* (C .* Omega) + ...
                  A .* ((Y(:, t-1) - mu) * (Y(:, t-1) - mu)') + B .* Sigma_t;
    end
end
