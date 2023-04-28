# Integrator

Numerical integrator written in Elixir for the solution of sets of non-stiff ordinary differential equations. 
Two integrator options are available; `ode45` which is an adaptation of the 
[Octave](https://octave.sourceforge.io/octave/function/ode45.html) and 
[Matlab](https://www.mathworks.com/help/matlab/ref/ode45.html).
`ode45` uses the Dormand-Prince 4th/5th order Runge Kutta algorithm.

`ode23` is an adaptation of the [Octave](https://octave.sourceforge.io/octave/function/ode23.html) and 
[Matlab](https://www.mathworks.com/help/matlab/ref/ode23.html) 
The `ode23` integrator uses the Bogacki-Shampine 3rd order Runge Kutta algorithm.

This library heavily leverages [Elixir Nx](https://github.com/elixir-nx/nx); many thanks to the
creators of `Nx`, as without it this library would not have been possible.

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

## Usage

See the Livebook guides for detailed examples of usage. As a simple example, you can integrate
the Van der pol equation as defined in `Integrator.Demo.van_der_pol_fn/2` via:

```elixir
t_initial = 0.0
t_final = 20.0
x_initial = Nx.tensor([0.0, 1.0])
solution = Integrator.integrate(&Demo.van_der_pol_fn/2, [t_initial, t_final], x_initial)
```

Then, `solution.output_t` contains a list of output times, and `solution.output_x` contains a list of 
values of `x` at these corresponding times.

Options exist for outputting dynamically via an output function, and also detecting events (such as
collisions); see the Livebooks for details.


