defmodule Integrator.RungeKutta.BogackiShampine23 do
  @moduledoc """
  Bogacki-Shampine
  """
  alias Integrator.{RungeKutta, Utils}
  @behaviour RungeKutta

  import Nx.Defn

  @impl RungeKutta
  def order, do: 3

  @impl RungeKutta
  def default_opts, do: []

  @impl RungeKutta
  defn integrate(ode_fn, t, x, dt, k_vals) do
  end

  @impl RungeKutta
  defn interpolate(t, x, der, t_out) do
    # Utils.hermite_quartic_interpolation(t, x, der, t_out)
  end
end
