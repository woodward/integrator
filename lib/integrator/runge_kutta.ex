defmodule Integrator.RungeKutta do
  @moduledoc """
  A behaviour that Runge-Kutta algorithms must implement. Currently, `Integrator.DormandPrince45`
  and `Integrator.BogackiShampine23` implement this behaviour.

  See the [list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
  """

  import Nx.Defn

  alias Integrator.AdaptiveStepsizeRefactor
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

  @doc """
  The core of the Runge-Kutta algorithm; computes the next step given a delta-t `dt` and all of the
  other constituent pieces.
  """
  @spec compute_step(
          stepper_fn :: stepper_fn_t(),
          ode_fn :: ode_fn_t(),
          t_old :: Nx.t(),
          x_old :: Nx.t(),
          k_vals_old :: Nx.t(),
          options_comp_old :: Nx.t(),
          dt :: Nx.t(),
          options :: AdaptiveStepsizeRefactor.NxOptions.t()
        ) :: {
          t_next :: Nx.t(),
          x_next :: Nx.t(),
          k_vals :: Nx.t(),
          options_comp :: Nx.t(),
          error :: Nx.t()
        }
  defn compute_step(stepper_fn, ode_fn, t_old, x_old, k_vals_old, options_comp_old, dt, options) do
    # This is the core of the Runge-Kutta algorithm for computing new the next step:
    {t_next, options_comp} = Utils.kahan_sum(t_old, options_comp_old, dt)
    {x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals_old, t_next)
    error = Utils.abs_rel_norm(x_next, x_old, x_est, options.abs_tol, options.rel_tol, options.norm_control?)
    {t_next, x_next, k_vals, options_comp, error}
  end
end
