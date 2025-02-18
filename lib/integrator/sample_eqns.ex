defmodule Integrator.SampleEqns do
  @moduledoc """
  Functions to be used in testing
  """
  import Nx.Defn

  @doc """
  From [octave](https://octave.sourceforge.io/octave/function/ode45.html) b


  fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
  """
  @spec van_der_pol_fn(Nx.t(), Nx.t()) :: Nx.t()
  defn van_der_pol_fn(_t, x) do
    x0 = x[0]
    x1 = x[1]

    one = Nx.tensor(1.0, type: Nx.type(x))
    new_x1 = Nx.subtract(one, Nx.pow(x0, 2)) |> Nx.multiply(x1) |> Nx.subtract(x0)
    Nx.stack([x1, new_x1])
  end

  @doc """
  The Euler equations of a rigid body without external forces. This is a standard test
  problem proposed by Krogh for solvers intended for nonstiff problems [see below].
  Based on "rigidode.m" from Matlab/Octave.  The analytical solutions are Jacobian
  elliptic functions.

  See [`rigidode.m`](http://www.ece.northwestern.edu/local-apps/matlabhelp/techdoc/math_anal/diffeq8.html)
  in Matlab.

  Shampine, L. F., and M. K. Gordon, Computer Solution of Ordinary Differential Equations,
  W.H. Freeman & Co., 1975
  """
  @spec euler_equations(Nx.t(), Nx.t()) :: Nx.t()
  defn euler_equations(_t, x) do
    # Octave:
    #   dxdt = [ x(2)*x(3) ;  -x(1)*x(3) ; -0.51*x(1)*x(2) ];
    x0 = x[1] * x[2]
    x1 = -x[0] * x[2]
    x2 = -0.51 * x[0] * x[1]
    Nx.stack([x0, x1, x2])
  end

  @acc_due_to_gravity -9.81

  @doc """
  Simulates a point mass or particle falling through pass affected by gravity.  Used for comparisons
  with the Matlab/Octave `ballode.m` routine
  """
  @spec falling_particle(Nx.t(), Nx.t()) :: Nx.t()
  defn falling_particle(_t, x) do
    x0 = x[1]
    x1 = Nx.tensor(@acc_due_to_gravity, type: Nx.type(x))
    Nx.stack([x0, x1])
  end

  @doc """
  Event function for a point mass or particle falling through pass affected by gravity.  Returns zero
  if x[0] is below the zero plane, or 1 if x[0] is above the zero plane.  Used for comparisons
  with the Matlab/Octave `ballode.m` routine
  """
  @spec falling_particle_event_fn(Nx.t(), Nx.t()) :: Nx.t()
  defn falling_particle_event_fn(_t, x) do
    # type = Nx.type(x)
    # zero = Nx.tensor(0, type: type)
    # one = Nx.tensor(1, type: type)
    if x[0] < 0.0, do: Nx.u8(0), else: Nx.u8(1)
  end
end
