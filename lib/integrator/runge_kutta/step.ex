defmodule Integrator.RungeKutta.Step do
  @moduledoc """
  The results of the computation of an individual Runge-Kutta step. `compute_step` computes
  the next Runge-Kutta step given info from the prior step.
  """

  import Nx.Defn

  alias Integrator.AdaptiveStepsizeRefactor.NxOptions
  alias Integrator.RungeKutta
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

  @spec initial_step(Nx.t(), Nx.t(), Keyword.t()) :: t()
  defn initial_step(t0, x0, opts \\ []) do
    opts = keyword!(opts, order: 5)
    type = Nx.type(x0)
    # The NaN values are just to allocate space for these values, and to also indicate that
    # for the moment they are not legitimate values (they will be swapped out later with computed values):
    nan = Nx.Constants.nan(type)

    %__MODULE__{
      t_old: nan,
      t_new: t0,
      #
      x_old: Nx.tensor(0.0, type: type) |> Nx.broadcast({Nx.size(x0)}),
      x_new: x0,
      #
      k_vals: initial_empty_k_vals_defn(x0, opts),
      options_comp: Nx.tensor(0.0, type: type),
      error_estimate: nan,
      dt: nan
    }
  end

  @spec interpolate_multiple_points(fun(), t(), NxOptions.t()) :: {Nx.t(), Nx.t()}
  defn interpolate_multiple_points(interpolate_fn, rk_step, options) do
    refine = options.refine
    type = options.type
    t_add = Nx.linspace(rk_step.t_old, rk_step.t_new, n: refine + 1, type: type)

    # Get rid of the first element (t_add[0]) via this slice:
    t_add = Nx.slice_along_axis(t_add, 1, refine, axis: 0)

    t = Nx.stack([rk_step.t_old, rk_step.t_new])
    x = Nx.stack([rk_step.x_old, rk_step.x_new]) |> Nx.transpose()
    x_out = interpolate_fn.(t, x, rk_step.k_vals, t_add)
    {t_add, x_out}
  end

  @spec interpolate_single_specified_point(fun(), t(), Nx.t()) :: {Nx.t(), Nx.t()}
  defn interpolate_single_specified_point(interpolate_fn, rk_step, t_add) do
    t = Nx.stack([rk_step.t_old, rk_step.t_new])
    x = Nx.stack([rk_step.x_old, rk_step.x_new]) |> Nx.transpose()
    interpolate_fn.(t, x, rk_step.k_vals, t_add) |> Utils.last_column()
  end

  @spec initial_empty_k_vals(integer(), Nx.t()) :: Nx.t()
  deftransform initial_empty_k_vals(order, x) do
    # Figure out the correct way to do this!  Does k_length depend on the order of the Runge Kutta method?
    k_length = order + 2

    x_length = Nx.size(x)
    zero = Nx.tensor(0.0, type: Nx.type(x))
    Nx.broadcast(zero, {x_length, k_length})
  end

  @spec initial_output_t_and_x(Nx.t(), NxOptions.t()) :: {Nx.t(), Nx.t()}
  deftransform initial_output_t_and_x(x0, options) do
    # I tried doing this function originally as a defn, but had problems getting the broadcast below to work; why???
    size_x = Nx.size(x0)
    zero = Nx.tensor(0.0, type: options.type)

    if options.fixed_output_times? == Nx.u8(1) or options.refine == 1 do
      t_ouptut = Nx.broadcast(zero, {1})
      x_output = Nx.broadcast(zero, {size_x})
      {t_ouptut, x_output}
    else
      add_points = options.refine
      t_ouptut = Nx.broadcast(zero, {add_points})
      x_output = Nx.broadcast(zero, {size_x, add_points})
      {t_ouptut, x_output}
    end
  end

  @spec initial_output_t_and_x_single_point(Nx.t(), NxOptions.t()) :: {Nx.t(), Nx.t()}
  deftransform initial_output_t_and_x_single_point(x0, options) do
    # I tried doing this function originally as a defn, but had problems getting the broadcast below to work; why???
    size_x = Nx.size(x0)
    zero = Nx.tensor(0.0, type: options.type)

    x_output = Nx.broadcast(zero, {size_x})
    {zero, x_output}
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
