function neglikelihood = garch_likelihood(Y, mu, Omega, alpha, beta, x, logLikelihoodFunc)
    [p, T] = size(Y);
    logL = 0;

    % Construct the matrices A, B, and C using the optimized x
    A_diag = alpha;
    B_diag = beta;
    C_diag = ones(p, 1); % Diagonal elements for C are ones

    A = Coeff_mat_a2b2(x(1), A_diag);
    B = Coeff_mat_a2b2(x(2), B_diag);
    C = Coeff_mat_a2b2(x(3), C_diag); % Default C using 1/sqrt(p)

    Sigma_t = Omega; % Initial covariance matrix

    for t = 2:T
        Sigma_t = (ones(p) - A - B) .* (C .* Omega) + ...
                  A .* ((Y(:, t-1) - mu) * (Y(:, t-1) - mu)') + B .* Sigma_t;
        
        logL = logL + sum(logLikelihoodFunc(Y(:, t), mu, Sigma_t));
    end

    % Regularization term
    neglikelihood = -logL;
end
