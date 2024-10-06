% Define the initial position and time driection
initial_conditions = [0; 1];
t_span = [0, 1.5708];
options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
harmonic_oscillator = @(t, X) [X(2); -X(1)];
[t, Y] = ode45(harmonic_oscillator, t_span, initial_conditions, options);

Y(end, :)