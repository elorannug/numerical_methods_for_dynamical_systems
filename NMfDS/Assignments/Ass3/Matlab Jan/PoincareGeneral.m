initial_conditions = [0; 1];
dir = 1;
step = 0.4;
t_span = dir * [0, step];
options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
harmonic_oscillator = @(t, X) [X(2); -X(1)];

numberOfCrossings = 3;
overallTime = 0;

for i = 1:3
    [newInitial, timeDuration] = PoincareMap(harmonic_oscillator, initial_conditions, dir, step, t_span);
    initial_conditions = newInitial;
    overallTime = overallTime + timeDuration;
end

overallTime


