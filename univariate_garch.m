function [alpha, beta] = univariate_garch(y, mu, sigma2, logLikelihoodFunc)
%     options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
    x0 = [0.1; 0.8]; % Initial guess for [alpha, beta]
    lb = [0; 0];
    ub = [1; 1];
    fun = @(x) univariate_instant_to_garch_likelihood(x, logLikelihoodFunc, y, mu, sigma2);

    x_opt = fmincon(fun, x0, [], [], [], [], lb, ub, @(x) garch_constraints(x));

    alpha = x_opt(1);
    beta = x_opt(2);
end

function garch_neg_llh = univariate_instant_to_garch_likelihood(param, logLikelihoodFunc, y, mu, sigma)
    a = param(1);
    b = param(2);
    T = length(y);
    garch_neg_llh = 0;
    sigma_t = sigma; % sigma here is actually sigma square

    for t = 2:T
        sigma_t = (1 - a - b) * sigma + a * (y(t-1) - mu)^2 + b * sigma_t;
        garch_neg_llh = garch_neg_llh + logLikelihoodFunc(y(t), mu, sigma_t); 
    end
    garch_neg_llh = -garch_neg_llh; % Negate to convert to negative log-likelihood
end
