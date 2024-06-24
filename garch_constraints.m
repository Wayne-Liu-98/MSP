function [c, ceq] = garch_constraints(x)
    % Ensure that alpha + beta < 1 for stationarity
    c = x(1) + x(2) - 1;
    ceq = [];
end