function logL = Gaussian_logLikelihoodFunc(y, mu, Sigma)
    logL = -0.5 * (log(det(Sigma)) + (y - mu)' * (Sigma \ (y - mu)) + length(y) * log(2 * pi));
end
