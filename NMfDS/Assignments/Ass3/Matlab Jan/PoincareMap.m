function [newInitial, TimeDuration] = PoincareMap(harmonic_oscillator, initial_conditions, dir, step, t_span)
options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% Procedure to compute when the x-axis is crossed
product = 1;
time = 0;
startPoint = initial_conditions;

while  product >= 0
    [t, Y] = ode45(harmonic_oscillator, t_span, initial_conditions, options);
    product = Y(1, 2) * Y(end, 2);
    initial_conditions = [Y(end, 1); Y(end, 2)];
    t_span = t_span + dir * step;
    time = time + step;
end

% Procedure to compute the exact time needed
for i = 1:100
    t_span = [0, time];
    [t, Y] = ode45(harmonic_oscillator, t_span, startPoint, options);
    scalar_product = -Y(end, 1);
    difference = Y(end, 2)/scalar_product;
    time = time - difference;    
    if abs(difference) < 1e-12
        break
    end
end
TimeDuration = time;
newInitial = [Y(end,1); Y(end,2)];