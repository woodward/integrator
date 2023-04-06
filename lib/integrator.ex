defmodule Integrator do
  @moduledoc """
  $$  F = M A $$

  or an inline equation via a single dollar sign, such as $ F = M A $

  Some more equations:

  $$ f(x) =  \\int_{-\\infty}^\\infty
    f\\hat(\\xi)\\,e^{2 \\pi i \\xi x}
    \\,d\\xi $$

  """
  def ode45(_deriv_fn, _x_initial, _x_final, _initial_y) do
    x = [0.0, 0.0180, 0.041]
    y = [Nx.tensor([2.0000, 0.000]), Nx.tensor([1.9897, -0.0322]), Nx.tensor([1.9979, -0.0638])]
    [x, y]
  end
end
