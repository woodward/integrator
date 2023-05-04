With Octave opts in ballode.m:

refine = 4;

% The options for solving the ordinary differential equation specify 
% to turn on event detection, and to make a plot of the output
options = odeset('Events',@events,'OutputSel',1,...
   'Refine',refine,'AbsTol',1.0e-14,'RelTol',1.0e-14);  
