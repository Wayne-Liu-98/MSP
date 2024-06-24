function coeff_mat = Coeff_mat_a2b2(var_2, mat_diag)
    % var_2 is 2^a or 2^b or 2^c, mat_diag is A_diag or B_diag or C_diag (ones),
    % The output is A or B or C
    p = length(mat_diag);
    var_1 = sqrt(mat_diag - (p-1) * (var_2^2));
    coeff_mat = var_2 * (var_1 + var_1') + (p-2) * var_2^2;
    coeff_mat(1:p+1:end) = mat_diag;
end
