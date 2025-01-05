defmodule Integrator.RungeKutta.Step do
  @moduledoc """
  The results of the computation of an individual Runge-Kutta step. `compute_step` computes
  the next Runge-Kutta step given info from the prior step.
  """

  import Nx.Defn

  alias Integrator.AdaptiveStepsizeRefactor.NxOptions
  alias Integrator.RungeKutta

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
  @spec compute_step(t(), Nx.t(), RungeKutta.stepper_fn_t(), RungeKutta.ode_fn_t(), NxOptions.t()) :: t()
  defn compute_step(step, dt, stepper_fn, ode_fn, options) do
    x_old = step.x_new
    t_old = step.t_new
    options_comp_old = step.options_comp
    k_vals_old = step.k_vals

    {t_new, x_new, k_vals_new, options_comp_new, error_estimate} =
      RungeKutta.compute_step(stepper_fn, ode_fn, t_old, x_old, k_vals_old, options_comp_old, dt, options)

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

  defn initial_step(t0, x0, opts \\ []) do
    opts = keyword!(opts, order: 5)
    type = Nx.type(x0)

    %__MODULE__{
      t_old: Nx.Constants.nan(type),
      t_new: t0,
      #
      x_old: Nx.Constants.nan(type),
      x_new: x0,
      #
      k_vals: initial_empty_k_vals_defn(x0, opts),
      options_comp: Nx.tensor(0.0, type: type),
      error_estimate: Nx.Constants.nan(type),
      dt: Nx.Constants.nan(type)
    }
  end

  @spec initial_empty_k_vals(integer(), Nx.t()) :: Nx.t()
  deftransform initial_empty_k_vals(order, x) do
    # Figure out the correct way to do this!  Does k_length depend on the order of the Runge Kutta method?
    k_length = order + 2

    x_length = Nx.size(x)
    zero = Nx.tensor(0.0, type: Nx.type(x))
    Nx.broadcast(zero, {x_length, k_length})
  end

  @spec initial_empty_k_vals_defn(Nx.t(), Keyword.t()) :: Nx.t()
  defn initial_empty_k_vals_defn(x, opts \\ []) do
    # Note that `order` needs to be passed in as an option, otherwise I get an error about a dimension
    # being a tensor if it's passed in as a standard argument
    opts = keyword!(opts, order: 5)

    # Figure out the correct way to do this!  Does k_length depend on the order of the Runge Kutta method?
    k_length = opts[:order] + 2

    x_length = Nx.size(x)
    zero = Nx.tensor(0.0, type: Nx.type(x))
    Nx.broadcast(zero, {x_length, k_length})
  end
end
