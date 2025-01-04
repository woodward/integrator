defmodule Integrator.RungeKutta.Step do
  @moduledoc """
  The results of the computation of an individual Runge-Kutta step. `compute_step` computes
  the next Runge-Kutta step given info from the prior step.
  """

  import Nx.Defn

  alias Integrator.AdaptiveStepsizeRefactor.NxOptions
  alias Integrator.RungeKutta
  alias Integrator.RungeKutta.Step
  alias Integrator.Utils
  alias Integrator.Utils

  @derive {Nx.Container,
   containers: [
     :t_old,
     :t_new,
     #
     :x_old,
     :x_new,
     #
     :k_vals,
     :options_comp,
     :error_estimate,
     :dt
   ]}

  @type t :: %__MODULE__{
          t_old: Nx.t(),
          t_new: Nx.t(),
          #
          x_old: Nx.t(),
          x_new: Nx.t(),
          #
          k_vals: Nx.t(),
          options_comp: Nx.t(),
          error_estimate: Nx.t(),
          dt: Nx.t()
        }

  defstruct [
    :t_old,
    :t_new,
    #
    :x_old,
    :x_new,
    #
    :k_vals,
    :options_comp,
    :error_estimate,
    :dt
  ]

  @doc """
  Computes the next `Runge-Kutta.Step` struct given a prior `RungeKutta.Step` which contains info from the
  previous Runge-Kutta computation
  """
  @spec compute_step(Step.t(), Nx.t(), RungeKutta.stepper_fn_t(), RungeKutta.ode_fn_t(), NxOptions.t()) :: Step.t()
  defn compute_step(step, dt, stepper_fn, ode_fn, options) do
    x_old = step.x_new
    t_old = step.t_new
    options_comp_old = step.options_comp
    k_vals_old = step.k_vals

    {t_new, x_new, k_vals_new, options_comp_new, error_estimate} =
      compute_step(stepper_fn, ode_fn, t_old, x_old, k_vals_old, options_comp_old, dt, options)

    %{
      step
      | t_old: t_old,
        x_old: x_old,
        #
        t_new: t_new,
        x_new: x_new,
        #
        k_vals: k_vals_new,
        options_comp: options_comp_new,
        error_estimate: error_estimate,
        dt: dt
    }
  end

  # Computes the next Runge-Kutta step and the associated error
  @spec compute_step(
          stepper_fn :: RungeKutta.stepper_fn_t(),
          ode_fn :: RungeKutta.ode_fn_t(),
          t_old :: Nx.t(),
          x_old :: Nx.t(),
          k_vals_old :: Nx.t(),
          options_comp_old :: Nx.t(),
          dt :: Nx.t(),
          options :: NxOptions.t()
        ) :: {
          t_next :: Nx.t(),
          x_next :: Nx.t(),
          k_vals :: Nx.t(),
          options_comp :: Nx.t(),
          error :: Nx.t()
        }
  defnp compute_step(stepper_fn, ode_fn, t_old, x_old, k_vals_old, options_comp_old, dt, options) do
    # This is the core of the Runge-Kutta algorithm for computing new the next step:
    {t_next, options_comp} = Utils.kahan_sum(t_old, options_comp_old, dt)
    {x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals_old, t_next)
    error = Utils.abs_rel_norm(x_next, x_old, x_est, options.abs_tol, options.rel_tol, options.norm_control?)
    {t_next, x_next, k_vals, options_comp, error}
  end
end
