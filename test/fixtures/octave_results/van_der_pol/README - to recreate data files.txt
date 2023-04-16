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

To simulate with event function for y(1) positive only:

function [value,isterminal,direction] = events(t,y)
  % FUNCTION events(t,y)    Event function for positive y(1)
  %   value = events(t,y) is a zero-crossing function that triggers an event
  %                       that may be used to halt simulation.
  %                       value is a vector of functions, the zero-crossing
  %                       of which is detected by the ODE solver.
  %
  %   [value, isterminal, direction] = events(t,y) can return isterminal = 1
  %                       to halt simulation, or 0 to restart.
  %                       direction = +1, 0, -1 specifies whether zero-crossings
  %                       should be detected in positive, any, or negative
  %                       directions.
  
  % Locate the time when y(1) passes through zero in a decreasing direction
  % and stop integration.
  value = y(1);     % detect height = 0
  isterminal = 1;   % stop the integration
  direction = -1;   % negative direction
end % subfunction

opts = odeset('Events',@events,'OutputSel',1,'Refine',4);  
fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
[t,y] = ode45 (fvdp, [0, 20], [2, 0], opts);


===========================================
To output values:

time_file_id = fopen('../test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/t.csv', 'w');
fdisp(time_file_id, t)
fclose(time_file_id)

x_file_id = fopen('../test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/x.csv', 'w');
fdisp(x_file_id, y)
fclose(x_file_id)

You'll then have to edit the file - change the blank space between the two values to ", "
Also "0" needs to be changed to "0.0"


