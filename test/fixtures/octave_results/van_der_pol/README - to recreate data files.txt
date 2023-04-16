format long   (to get extra precision in the output fles)

fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
[t,y] = ode45 (fvdp, [0, 20], [2, 0]);

===========================================

To simulate with more precision for "time_high_fidelity.csv" and "x_high_fidelity.csv"
fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];

opts = odeset("AbsTol", 1.0e-10, "RelTol", 1.0e-10)
[t,y] = ode45 (fvdp, [0, 20], [2, 0], opts);

===========================================

To simulate with interpolation turned off (i.e., refine = 1):
fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];

opts = odeset("Refine", 1)
[t,y] = ode45 (fvdp, [0, 20], [2, 0], opts);

===========================================
To output values:

time_file_id = fopen('../test/fixtures/integrator/integrator/time_high_fidelity.csv', 'w');
fdisp(time_file_id, t)
fclose(time_file_id)

x_file_id = fopen('../test/fixtures/integrator/integrator/x_high_fidelity.csv', 'w');
fdisp(x_file_id, y)
fclose(x_file_id)

You'll then have to edit the file - change the blank space between the two values to ", "
Also "0" needs to be changed to "0.0"


