function [c, ceq] = positive_def_constraint(x, Omega, alpha, beta, p)
    A_diag = alpha;
    B_diag = beta;
    C_diag = ones(p, 1); % Diagonal elements for C are ones

    A = Coeff_mat_a2b2(x(1), A_diag);
    B = Coeff_mat_a2b2(x(2), B_diag);
    C = Coeff_mat_a2b2(x(3), C_diag); % Using x(3) for C

    % Ensure (ones(p) * ones(p)' - A - B) .* C .* Omega is positive definite
    M = (ones(p) * ones(p)' - A - B) .* (C .* Omega);

    % Calculate the eigenvalues of M
    eigenvalues = eig(M);

    % The constraint is that all eigenvalues should be positive
    c = -min(eigenvalues); % Ensure all eigenvalues are positive
    ceq = [];
end
