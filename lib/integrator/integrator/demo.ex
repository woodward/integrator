defmodule Integrator.Demo do
  @moduledoc """
  Functions to be used in testing
  """
  import Nx.Defn

  @doc """
  From [octave](https://octave.sourceforge.io/octave/function/ode45.html) but with
  the decrementing the indices by one

  x(1) => x(0) and x(2) => x(1):
  fvdp = @(t,x) [x(1); (1 - x(0)^2) * x(1) - x(0)];
  """
  defn van_der_pol_fn(_t, x) do
    x0 = x[0]
    x1 = x[1]

    one = Nx.tensor(1.0, type: Nx.type(x))
    new_x1 = Nx.subtract(one, Nx.pow(x0, 2)) |> Nx.multiply(x1) |> Nx.subtract(x0)
    Nx.stack([x1, new_x1])
  end
end
