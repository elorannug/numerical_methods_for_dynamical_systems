% Define the initial position and time driection
initial_conditions = [0; 1];
dir = 1;
step = 0.4;
t_span = dir * [0, step];
options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
harmonic_oscillator = @(t, X) [X(2); -X(1)];

% Procedure to compute when the x-axis is crossed
product = 1;
time = 0;
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
    initial_conditions = [0; 1];
    [t, Y] = ode45(harmonic_oscillator, t_span, initial_conditions, options);
    scalar_product = Y(end, 1);
    difference = Y(end, 2)/(-Y(end, 1));
    time = time - difference;    
    if abs(difference) < 1e-12
        break
    end
end
time
