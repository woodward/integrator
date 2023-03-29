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

Using ode45:
https://www.eng.auburn.edu/~tplacek/courses/3600/ode45berkley.pdf
