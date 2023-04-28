# Integrator

A numerical integrator written in Elixir for the solution of sets of non-stiff ordinary differential
equations (ODEs). 

## Installation

The package can be installed by adding `integrator` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:integrator, "~> 0.1"},
  ]
end
```

The docs can be found at <https://hexdocs.pm/integrator>.

## Description

Two integrator options are available; `ode45` which is an adaptation of the [Octave
ode45](https://octave.sourceforge.io/octave/function/ode45.html) and [Matlab
ode45](https://www.mathworks.com/help/matlab/ref/ode45.html). The `ode45` integrator utilizes the
[Dormand-Prince](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method) 4th/5th order Runge
Kutta algorithm.

`ode23` is an adaptation of the [Octave
ode23](https://octave.sourceforge.io/octave/function/ode23.html) and [Matlab
ode23](https://www.mathworks.com/help/matlab/ref/ode23.html) The `ode23` integrator uses the
[Bogacki-Shampine](https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method) 3rd order Runge
Kutta algorithm.

Both `ode45` (which is the default integrator option) and `ode23` utilize an adaptive stepsize
algorithm for computing the integration time step.  The time step is computed based on the
satisfaction of a required error tolerance.

This library heavily leverages [Elixir Nx](https://github.com/elixir-nx/nx); many thanks to the
[creators of `Nx`](https://github.com/elixir-nx/nx/graphs/contributors), as without it this library
would not have been possible. The [GNU Octave code](https://github.com/gnu-octave/octave) was also
used heavily for inspiration and was used to generate numerical test cases for the Elixir versions
of the algorithms.  Many thanks to [John W. Eaton](https://jweaton.org/) for his tremendous work on
Octave. `Integrator` has been tested extensively during its development, and has a large and growing
test suite.

## Usage

See the Livebook guides for detailed examples of usage. As a simple example, you can integrate the
Van der Pol equation as defined in `Integrator.Demo.van_der_pol_fn/2` from time 0 to 20 with an
intial x value of `[0, 1]` via:

```elixir
t_initial = 0.0
t_final = 20.0
x_initial = Nx.tensor([0.0, 1.0])
solution = Integrator.integrate(&Demo.van_der_pol_fn/2, [t_initial, t_final], x_initial)
```

Then, `solution.output_t` contains a list of output times, and `solution.output_x` contains a list
of values of `x` at these corresponding times.

Options exist for:
- outputting simulation results dynamically via an output function (for applications
such as plotting dynamically, or for animating while the simulation is underway)
- generating simulation output at fixed times (such as at `t = 0.1, 0.2, 0.3`, etc.)
- interpolating intermediate points via quartic Hermite interpolation (for `ode45`) or via cubic
Hermite interpolation (for `ode23`) 
- detecting termination events (such as collisions); see the Livebooks for details.
- increasing the simulation fidelity (at the expense of simulation time) via absolute tolerance and
  relative tolerance settings


