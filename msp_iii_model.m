function [A, B, C, mu, Omega] = msp_iii_model(Y, logLikelihoodFunc, LASSO_lambda)
    if nargin < 3
        LASSO_lambda = 0;
    end
    
    [p, T] = size(Y);

    % Step 1: Estimate the unconditional parameters
    mu = mean(Y, 2);
    Omega = cov(Y');

    % Step 2: Estimate univariate GARCH parameters for each series
    alpha = zeros(p, 1);
    beta = zeros(p, 1);
    sigma2 = var(Y, 0, 2);
    
    for i = 1:p
        [alpha(i), beta(i)] = univariate_garch(Y(i,:), mu(i), sigma2(i), logLikelihoodFunc);
    end

    % Step 3: Optimize for A, B, and C matrices
    options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
    x0 = zeros(3,1); % Initial guess for [2^a, 2^b]
    lb = [-sqrt(min(alpha)/(p-1)); -sqrt(min(beta)/(p-1)); -sqrt(1/(p-1))];
    ub = [sqrt(min(alpha)/(p-1)); sqrt(min(beta)/(p-1)); sqrt(1/(p-1))];
    A = zeros(p);
    B = zeros(p);
    C = eye(p); % Initialize C as identity matrix

    fun = @(x) garch_likelihood(Y, mu, Omega, alpha, beta, x, logLikelihoodFunc)...
        + LASSO_lambda*abs(x(3)-1/sqrt(p)); % LASSO punishment for C to go away from 1 

    x_opt = fmincon(fun, x0, [], [], [], [], lb, ub, @(x) positive_def_constraint(x, Omega, alpha, beta, p));

    % Construct the matrices A, B, and C using the optimized x
    A = Coeff_mat_a2b2(x_opt(1), alpha);
    B = Coeff_mat_a2b2(x_opt(2), beta);
    C = Coeff_mat_a2b2(x_opt(3), ones(p, 1)); % Using x(3) for C
end
