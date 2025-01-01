defmodule Integrator.RungeKutta do
  @moduledoc """
  A behaviour that Runge-Kutta algorithms must implement. Currently, `Integrator.DormandPrince45`
  and `Integrator.BogackiShampine23` implement this behaviour.

  See the [list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
  """

  alias Integrator.Utils

  @type stepper_fn_t :: ((Nx.t(), Nx.t() -> Nx.t()), Nx.t(), Nx.t(), Nx.t(), Nx.t(), Nx.t() -> {Nx.t(), Nx.t(), Nx.t(), Nx.t()})
  @type ode_fn_t :: (Nx.t(), Nx.t() -> Nx.t())
  @type interpolate_fn_t :: (Nx.t(), Nx.t(), Nx.t(), Nx.t() -> Nx.t())

  @doc """
  Integrates an ODE function
  """
  @callback integrate(
              ode_fn :: ode_fn_t(),
              t :: Nx.t(),
              x :: Nx.t(),
              dt :: Nx.t(),
              k_vals :: Nx.t(),
              t_next :: Nx.t()
            ) ::
              {x_next :: Nx.t(), x_est :: Nx.t(), k_new :: Nx.t()}

  @doc """
  Interpolates using the method that is suitable for this particular Runge-Kutta method
  """
  @callback interpolate(t :: Nx.t(), x :: Nx.t(), der :: Nx.t(), t_out :: Nx.t()) :: x_out :: Nx.t()

  @doc """
  The order of this Runge-Kutta method
  """
  @callback order() :: integer()

  # Convert to defn:
  def interpolate(t_add, interpolate_fn, t_old, x_old, t_new, x_new, k_vals) do
    t_add_length =
      case Nx.shape(t_add) do
        {} -> 1
        {length} -> length
      end

    t = Nx.stack([t_old, t_new])
    x = Nx.stack([x_old, x_new]) |> Nx.transpose()

    x_out = interpolate_fn.(t, x, k_vals, t_add)
    x_out |> Utils.columns_as_list(0, t_add_length - 1)
  end
end
