Files were copied from:
https://github.com/gnu-octave/octave/tree/default/scripts/ode

and were flattened into this directory

2023-03-28

ode45 documentation:
https://octave.sourceforge.io/octave/function/ode45.html

Matlab ode45 documentation:
https://www.mathworks.com/help/matlab/ref/ode45.html


Example: Solve the Van der Pol equation

fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
[t,y] = ode45 (fvdp, [0, 20], [2, 0]);

plot(t,y(:,1),'-o',t,y(:,2),'-o')
title('Solution of van der Pol Equation (\mu = 1) with ODE45');
xlabel('Time t');
ylabel('Solution y');
legend('y_1','y_2')

-------------

tspan = [0 5];
y0 = 0;
[t,y] = ode45(@(t,y) 2*t, tspan, y0);
Plot the solution.

plot(t,y,'-o')


Using ode45:
https://www.eng.auburn.edu/~tplacek/courses/3600/ode45berkley.pdf

==================================================================================================
Visual representation of octave call stack:

ode45
  odeset                             (ode45:                         116)
  odedefaults                        (ode45:                         160)
  odeset                             (ode45:                         163)
  odemergeopts                       (ode45:                         173) 
  starting_stepsize                  (ode45:                         199)
    AbsRel_norm                        (starting_stepsize:            50)
    AbsRel_norm                        (starting_stepsize:            57)
    AbsRel_norm                        (starting_stepsize:            73)
  integrate_adaptive                 (ode45:                         239)
    init event handler (optional)      (integrate_adaptive:          122)
    <main simulation loop>             (integrate_adaptive:          143)
    ---------------------------------
    kahan                                (integrate_adaptive:        146)
    runge_kutta_45_dorpri                (integrate_adaptive:        147)
    AbsRel_norm                          (integrate_adaptive:        157)
    ode_event_handler (optional)         (integrate_adaptive:        174)
    runge_kutta_interpolate              (integrate_adaptive:        211)
      hermite_quartic_interpolation        (runge_kutta_interpolate:  50)
    call output function                 (integrate_adaptive:        224)
    compute dt                           (integrate_adaptive:        267)
    ---------------------------------
    kahan                                (integrate_adaptive:        146)
    runge_kutta_45_dorpri                (integrate_adaptive:        147)
    AbsRel_norm                          (integrate_adaptive:        157)
    runge_kutta_interpolate              (integrate_adaptive:        211)
      hermite_quartic_interpolation        (runge_kutta_interpolate:  50)
    call output function                 (integrate_adaptive:        224)
    compute dt                           (integrate_adaptive:        267)
    ---------------------------------
    ...
    not checking if integration was successful
    integrate_adaptive -- end of function    
    