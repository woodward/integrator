# GNU Octave Code

This code is NOT USED in any way in the Elixir Integrator library; it is just included 
here purely as a reference, and is also used to generate the Octave test data (see test/fixtures/octave_results).

The requisite files to run ode45 and ode23 were copied from:
https://github.com/gnu-octave/octave/tree/default/scripts/ode

and were flattened into this directory.

Octave ode45 & ode23 documentation:
https://octave.sourceforge.io/octave/function/ode45.html
https://octave.sourceforge.io/octave/function/ode23.html

Matlab ode45 & ode23 documentation:
https://www.mathworks.com/help/matlab/ref/ode45.html
https://www.mathworks.com/help/matlab/ref/ode23.html


## Example: Solve the Van der Pol equation

fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
[t,x] = ode45 (fvdp, [0, 20], [2, 0]);
OR
[t,x] = ode23 (fvdp, [0, 20], [2, 0]);
[t,x] = ode23 (fvdp, [0, 20], [2, 0], odeset ("Refine", 4));

To simulate with more precision:  
opts = odeset("AbsTol", 1.0e-10, "RelTol", 1.0e-10)
[t,x] = ode45 (fvdp, [0, 20], [2, 0], opts);

plot(t,x(:,1),'-o',t,x(:,2),'-o')
title('Solution of van der Pol Equation (\mu = 1) with ODE45');
xlabel('Time t');
ylabel('Solution x');
legend('x_1','x_2')

-------------

tspan = [0 5];
x0 = 0;
[t,x] = ode45(@(t,x) 2*t, tspan, x0);
Plot the solution.

plot(t,x,'-o')


Some notes on the usage of Octave's ode45:
https://www.eng.auburn.edu/~tplacek/courses/3600/ode45berkley.pdf

To write values to a file [see here](https://en.wikibooks.org/wiki/Octave_Programming_Tutorial/Text_and_file_output)
format long   (to get extra precision)
file_id = fopen('mydata.txt', 'w');

Then:
fdisp(file_id, value)
or
fprintf(file_id, '%f\n', value)
fprintf(file_id, '%d\n', value)

fclose(file_id)


---------------------------------------

## Visual representation of the octave call stack when invoking ode45:

ode45
  odeset                                 (ode45:                         116)
  odedefaults                            (ode45:                         160)
  odeset                                 (ode45:                         163)
  odemergeopts                           (ode45:                         173) 
  starting_stepsize                      (ode45:                         199)
    AbsRel_norm                            (starting_stepsize:            50)
    AbsRel_norm                            (starting_stepsize:            57)
    AbsRel_norm                            (starting_stepsize:            73)
  integrate_adaptive                     (ode45:                         239)
    init event handler (optional)          (integrate_adaptive:          122)
    <main integration loop>                (integrate_adaptive:          143)
      ---------------------------------        
      kahan                                  (integrate_adaptive:        146)
      runge_kutta_45_dorpri                  (integrate_adaptive:        147)
      AbsRel_norm                            (integrate_adaptive:        157)
      call ode_event_handler (optional)      (integrate_adaptive:        174)
      runge_kutta_interpolate                (integrate_adaptive:        211)
        hermite_quartic_interpolation          (runge_kutta_interpolate:  50)
      call output function                   (integrate_adaptive:        224)
      compute dt                             (integrate_adaptive:        267)
      ---------------------------------        
      kahan                                  (integrate_adaptive:        146)
      runge_kutta_45_dorpri                  (integrate_adaptive:        147)
      AbsRel_norm                            (integrate_adaptive:        157)
      runge_kutta_interpolate                (integrate_adaptive:        211)
        hermite_quartic_interpolation          (runge_kutta_interpolate:  50)
      call output function                   (integrate_adaptive:        224)
      compute dt                             (integrate_adaptive:        267)
      ---------------------------------        
      ...    
    -------- end of integration loop       (integrate_adaptive:          277)
    checking if successful                 (integrate_adaptive:          280)
    integrate_adaptive -- end              (integrate_adaptive:          299)      
    