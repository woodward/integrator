defmodule Integrator.RungeKutta.Step do
  @moduledoc """
  The results of the computation of an individual Runge-Kutta step
  """

  @derive {Nx.Container,
   containers: [
     :t_old,
     :t_new,
     :dt,
     #
     :x_old,
     :x_new,
     #
     :k_vals,
     :options_comp,
     :error_estimate
   ]}

  @type t :: %__MODULE__{
          t_old: Nx.t(),
          t_new: Nx.t(),
          dt: Nx.t(),
          #
          x_old: Nx.t(),
          x_new: Nx.t(),
          #
          k_vals: Nx.t(),
          options_comp: Nx.t(),
          error_estimate: Nx.t()
        }

  defstruct [
    :t_old,
    :t_new,
    :dt,
    #
    :x_old,
    :x_new,
    #
    :k_vals,
    :options_comp,
    :error_estimate
  ]

  import Nx.Defn

  alias Integrator.RungeKutta
  alias Integrator.RungeKutta.Step
  alias Integrator.Utils

  # Computes the next Runge-Kutta step. Note that this function "wraps" the Nx functions which
  # perform the actual numerical computations
  @spec compute_step(Step.t(), RungeKutta.stepper_fn_t(), RungeKutta.ode_fn_t(), Keyword.t()) :: Step.t()
  def compute_step(step, stepper_fn, ode_fn, opts) do
    x_old = step.x_new
    t_old = step.t_new
    options_comp_old = step.options_comp
    k_vals_old = step.k_vals
    dt = step.dt

    {t_next, x_next, k_vals, options_comp, error_estimate} =
      compute_step_nx(stepper_fn, ode_fn, t_old, x_old, k_vals_old, options_comp_old, dt, opts)

    %Step{
      t_new: t_next,
      x_new: x_next,
      k_vals: k_vals,
      options_comp: options_comp,
      error_estimate: error_estimate
    }
  end

  # Computes the next Runge-Kutta step and the associated error
  @spec compute_step_nx(
          stepper_fn :: RungeKutta.stepper_fn_t(),
          ode_fn :: RungeKutta.ode_fn_t(),
          t_old :: Nx.t(),
          x_old :: Nx.t(),
          k_vals_old :: Nx.t(),
          options_comp_old :: Nx.t(),
          dt :: Nx.t(),
          opts :: Keyword.t()
        ) :: {
          t_next :: Nx.t(),
          x_next :: Nx.t(),
          k_vals :: Nx.t(),
          options_comp :: Nx.t(),
          error :: Nx.t()
        }
  defnp compute_step_nx(stepper_fn, ode_fn, t_old, x_old, k_vals_old, options_comp_old, dt, opts) do
    {t_next, options_comp} = Utils.kahan_sum(t_old, options_comp_old, dt)
    {x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals_old, t_next)
    error = abs_rel_norm(x_next, x_old, x_est, opts[:abs_tol], opts[:rel_tol], norm_control: opts[:norm_control])
    {t_next, x_next, k_vals, options_comp, error}
  end

  # Originally based on
  # [Octave function AbsRelNorm](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/AbsRel_norm.m)

  # Options
  # * `:norm_control` - Control error relative to norm; i.e., control the error `e` at each step using the norm of the
  #   solution rather than its absolute value.  Defaults to true.

  # See [Matlab documentation](https://www.mathworks.com/help/matlab/ref/odeset.html#bu2m9z6-NormControl)
  # for a description of norm control.
  @spec abs_rel_norm(Nx.t(), Nx.t(), Nx.t(), float(), float(), Keyword.t()) :: Nx.t()
  defn abs_rel_norm(t, t_old, x, abs_tolerance, rel_tolerance, opts \\ []) do
    if opts[:norm_control] do
      # Octave code
      # sc = max (AbsTol(:), RelTol * max (sqrt (sumsq (t)), sqrt (sumsq (t_old))));
      # retval = sqrt (sumsq ((t - x))) / sc;

      max_sq_t = Nx.max(Nx.LinAlg.norm(t), Nx.LinAlg.norm(t_old))
      sc = Nx.max(abs_tolerance, rel_tolerance * max_sq_t)
      Nx.LinAlg.norm(t - x) / sc
    else
      # Octave code:
      # sc = max (AbsTol(:), RelTol .* max (abs (t), abs (t_old)));
      # retval = max (abs (t - x) ./ sc);

      sc = Nx.max(abs_tolerance, rel_tolerance * Nx.max(Nx.abs(t), Nx.abs(t_old)))
      (Nx.abs(t - x) / sc) |> Nx.reduce_max()
    end
  end
end
