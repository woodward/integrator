defmodule Integrator.Test do
  @moduledoc """
  Functions to be used in testing
  """

  @doc """
  From [octave](https://octave.sourceforge.io/octave/function/ode45.html) but with
  the translations:

  y(1) => y(0) and y(2) => y(1):
  fvdp = @(t,y) [y(1); (1 - y(0)^2) * y(1) - y(0)];
  """
  def van_der_pol_fn(_t, y) do
    y0 = y[0]
    y1 = y[1]

    one = Nx.tensor(1.0, type: Nx.type(y))
    new_y1 = Nx.subtract(one, Nx.pow(y0, 2)) |> Nx.multiply(y1) |> Nx.subtract(y0)
    Nx.stack([y1, new_y1])
  end
end
