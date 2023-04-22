defmodule Integrator.AdaptiveStepsize do
  @moduledoc false
  import Nx.Defn
  alias Integrator.{MaxErrorsExceededError, NonlinearEqnRoot, Utils}

  defmodule ComputedStep do
    @moduledoc false
    defstruct [
      :t_new,
      :x_new,
      :k_vals,
      :options_comp
    ]
  end

  @type t :: %__MODULE__{
          t_old: Nx.t() | nil
          # :x_old,
          # #
          # :t_new,
          # :x_new,
          # #
          # :t_new_rk_interpolate,
          # :x_new_rk_interpolate,
          # #
          # :dt,
          # :k_vals,
          # #
          # options_comp: 0.0,
          # #
          # count_loop__increment_step: 0,
          # count_cycles__compute_step: 0,
          # count_save: 2,
          # #
          # # ireject in Octave:
          # error_count: 0,
          # i_step: 0,
          # #
          # unhandled_termination: true,
          # terminal_event: :continue,
          # terminal_output: :continue,
          # #
          # # The output of the integration:
          # ode_t: [],
          # ode_x: [],
          # #
          # # The output of the integration, including the interpolated points:
          # output_t: [],
          # output_x: [],
          # #
          # # The last chunk of points; will include the computed point plus the interpolated points (if
          # # interpolation is enabled) or just the computed point (if interpolation is disabled):
          # t_new_chunk: [],
          # x_new_chunk: []
        }
  defstruct [
    :t_old,
    :x_old,
    #
    :t_new,
    :x_new,
    #
    :t_new_rk_interpolate,
    :x_new_rk_interpolate,
    #
    :dt,
    :k_vals,
    #
    options_comp: 0.0,
    #
    count_loop__increment_step: 0,
    count_cycles__compute_step: 0,
    count_save: 2,
    #
    # ireject in Octave:
    error_count: 0,
    i_step: 0,
    #
    unhandled_termination: true,
    terminal_event: :continue,
    terminal_output: :continue,
    #
    # The output of the integration:
    ode_t: [],
    ode_x: [],
    #
    # The output of the integration, including the interpolated points:
    output_t: [],
    output_x: [],
    #
    # The last chunk of points; will include the computed point plus the interpolated points (if
    # interpolation is enabled) or just the computed point (if interpolation is disabled):
    t_new_chunk: [],
    x_new_chunk: []
  ]

  defmodule MaxErrorsExceededError do
    defexception message: "Too many errors",
                 error_count: 0,
                 max_number_of_errors: 0,
                 step: nil
  end

  @stepsize_factor_min 0.8
  @stepsize_factor_max 1.5

  @default_refine 4
  @default_max_number_of_errors 5_000
  @default_max_step 2.0
  @default_store_resuts true

  # Switch to using Utils.epsilon/1
  @epsilon 2.2204e-16

  @nx_true Nx.tensor(1, type: :u8)
  # @nx_false Nx.tensor(0, type: :u8)

  @spec default_opts() :: Keyword.t()
  def default_opts() do
    [
      epsilon: @epsilon,
      max_number_of_errors: @default_max_number_of_errors,
      max_step: @default_max_step,
      refine: @default_refine,
      store_resuts?: @default_store_resuts
    ]
  end

  @doc """

  See [Wikipedia](https://en.wikipedia.org/wiki/Adaptive_stepsize)
  """
  @spec integrate(fun(), fun(), fun(), [Nx.t() | float()], float, Nx.t(), integer(), Keyword.t()) :: t()
  def integrate(stepper_fn, interpolate_fn, ode_fn, [t_start, t_end], initial_tstep, x0, order, opts \\ []) do
    opts = default_opts() |> Keyword.merge(Utils.default_opts()) |> Keyword.merge(opts)
    if fun = opts[:output_fn], do: fun.([t_start], [x0])

    %__MODULE__{
      t_new: t_start,
      x_new: x0,
      dt: initial_tstep,
      k_vals: initial_empty_k_vals(order, x0)
    }
    |> store_first_point(t_start, x0, opts[:store_resuts?])
    |> step_forward(t_start, t_end, :continue, stepper_fn, interpolate_fn, ode_fn, order, opts)
    |> reverse_results()
  end

  @doc """
  Implements the Kahan summation algorithm, also known as compensated summation.
  Based on this [code in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/kahan.m).
  This is really a private function, but is made public so the docs are visible.

  The algorithm significantly reduces the numerical error in the total
  obtained by adding a sequence of finite precision floating point numbers
  compared to the straightforward approach.  For more details
  see [this Wikipedia entry](http://en.wikipedia.org/wiki/Kahan_summation_algorithm).
  This function is called by AdaptiveStepsize.integrate to better catch
  equality comparisons.

  The first input argument is the variable that will contain the summation.
  This variable is also returned as the first output argument in order to
  reuse it in subsequent calls to `Integrator.AdaptiveStepsize.kahan_sum/3` function.

  The second input argument contains the compensation term and is returned
  as the second output argument so that it can be reused in future calls of
  the same summation.

  The third input argument `term` is the variable to be added to `sum`.
  """
  defn kahan_sum(sum, comp, term) do
    # Octave code:
    # y = term - comp;
    # t = sum + y;
    # comp = (t - sum) - y;
    # sum = t;

    y = term - comp
    t = sum + y
    comp = t - sum - y
    sum = t

    {sum, comp}
  end

  defp store_first_point(step, t_start, x0, true = _store_resuts?) do
    %{step | output_t: [t_start], output_x: [x0], ode_t: [t_start], ode_x: [x0]}
  end

  defp store_first_point(step, _t_start, _x0, _store_resuts?), do: step

  defp initial_empty_k_vals(order, x) do
    # Figure out the correct way to do this!  Does k_length depend on the order of the Runge Kutta method?
    k_length = order + 2

    {length_x} = Nx.shape(x)
    Nx.broadcast(0.0, {length_x, k_length})
  end

  defp step_forward(step, t_old, t_end, _halt?, _stepper_fn, _interpolate_fn, _ode_fn, _order, _opts) when t_old >= t_end do
    step
  end

  defp step_forward(step, _t_old, _t_end, :halt, _stepper_fn, _interpolate_fn, _ode_fn, _order, _opts) do
    step
  end

  defp step_forward(step, _t_old, t_end, _halt?, stepper_fn, interpolate_fn, ode_fn, order, opts) do
    {new_step, error_est} = compute_step(step, stepper_fn, ode_fn, opts)
    step = step |> increment_compute_counter()

    step =
      if less_than_one?(error_est) do
        step
        |> increment_and_reset_counters()
        |> merge_new_step(new_step)
        |> call_event_fn(opts[:event_fn], interpolate_fn, opts)
        |> interpolate(interpolate_fn, opts[:refine])
        |> store_resuts(opts[:store_resuts?])
        |> call_output_fn(opts[:output_fn])
      else
        bump_error_count(step, opts)
      end

    dt = compute_next_timestep(step.dt, Nx.to_number(error_est), order, Nx.to_number(step.t_new), t_end, opts)
    step = %{step | dt: dt}

    step
    |> step_forward(t_next(step, dt), t_end, halt?(step), stepper_fn, interpolate_fn, ode_fn, order, opts)
  end

  defp less_than_one?(error_est), do: Nx.less(error_est, 1.0) == @nx_true

  defp halt?(%{terminal_event: :halt} = _step), do: :halt
  defp halt?(%{terminal_output: :halt} = _step), do: :halt
  defp halt?(_step), do: :continue

  defp bump_error_count(step, opts) do
    step = %{step | error_count: step.error_count + 1}

    if step.error_count > opts[:max_number_of_errors] do
      raise MaxErrorsExceededError,
        message: "Too many errors",
        error_count: step.error_count,
        max_number_of_errors: opts[:max_number_of_errors],
        step: step
    end

    step
  end

  defp t_next(%{error_count: error_count} = step, dt) when error_count > 0 do
    # Update this into step somehow???
    Nx.to_number(step.t_old) + dt
  end

  defp t_next(%{error_count: error_count} = step, _dt) when error_count == 0 do
    Nx.to_number(step.t_new)
  end

  defp reverse_results(step) do
    %{
      step
      | output_x: step.output_x |> Enum.reverse(),
        output_t: step.output_t |> Enum.reverse(),
        #
        ode_x: step.ode_x |> Enum.reverse(),
        ode_t: step.ode_t |> Enum.reverse()
    }
  end

  # Formula taken from Hairer
  defp compute_next_timestep(dt, error, order, t_old, t_end, opts) do
    # Avoid divisions by zero:
    error = error + opts[:epsilon]

    # factor should be cached somehow; perhaps passed in in the options?
    factor = Math.pow(0.38, 1.0 / (order + 1))

    foo = factor * Math.pow(1 / error, 1 / (order + 1))

    dt = dt * min(@stepsize_factor_max, max(@stepsize_factor_min, foo))
    dt = min(abs(dt), opts[:max_step])

    # ## Make sure we don't go past t_end:
    min(abs(dt), abs(t_end - t_old))
  end

  defp increment_and_reset_counters(step) do
    %{
      step
      | count_loop__increment_step: step.count_loop__increment_step + 1,
        i_step: step.i_step + 1,
        error_count: 0
        # terminal_event: false,
        # terminal_output: false
    }
  end

  defp merge_new_step(step, computed_step) do
    %{
      step
      | x_old: step.x_new,
        t_old: step.t_new,
        x_new: computed_step.x_new,
        t_new: computed_step.t_new,
        x_new_rk_interpolate: computed_step.x_new,
        t_new_rk_interpolate: computed_step.t_new,
        k_vals: computed_step.k_vals,
        options_comp: computed_step.options_comp
    }
  end

  defp store_resuts(step, false = _store_resuts?) do
    step
  end

  defp store_resuts(step, true = _store_resuts?) do
    %{
      step
      | ode_t: [step.t_new | step.ode_t],
        ode_x: [step.x_new | step.ode_x],
        output_t: step.t_new_chunk ++ step.output_t,
        output_x: step.x_new_chunk ++ step.output_x
    }
  end

  defp increment_compute_counter(step) do
    %{step | count_cycles__compute_step: step.count_cycles__compute_step + 1}
  end

  defp compute_step(step, stepper_fn, ode_fn, opts) do
    x_old = step.x_new
    t_old = step.t_new
    options_comp_old = step.options_comp
    k_vals = step.k_vals
    dt = step.dt

    {_t_new, options_comp} = kahan_sum(t_old, options_comp_old, dt)
    {t_next, x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals)

    # Pass these in as options:
    norm_control = false
    error = Utils.abs_rel_norm(x_next, x_old, x_est, opts[:abs_tol], opts[:rel_tol], norm_control: norm_control)

    {%ComputedStep{
       x_new: x_next,
       t_new: t_next,
       k_vals: k_vals,
       options_comp: options_comp
     }, error}
  end

  defp interpolate(step, _interpolate_fn, refine) when refine == 1 do
    %{step | t_new_chunk: [Nx.to_number(step.t_new)], x_new_chunk: [step.x_new]}
  end

  defp interpolate(step, interpolate_fn, refine) when refine > 1 do
    tadd = Nx.linspace(step.t_old, step.t_new, n: refine + 1, type: Nx.type(step.x_old))
    # Get rid of the first element (tadd[0]) via this slice:
    tadd = Nx.slice_along_axis(tadd, 1, refine, axis: 0)

    x_out_as_cols = do_interpolation(step, interpolate_fn, tadd)

    %{step | x_new_chunk: x_out_as_cols |> Enum.reverse(), t_new_chunk: Nx.to_list(tadd) |> Enum.reverse()}
  end

  defp interpolate_one_point(t_new, step, interpolate_fn) do
    do_interpolation(step, interpolate_fn, Nx.tensor(t_new)) |> List.first()
  end

  defp do_interpolation(step, interpolate_fn, tadd) do
    tadd_length =
      case Nx.shape(tadd) do
        {} -> 1
        {length} -> length
      end

    t = Nx.stack([step.t_old, step.t_new_rk_interpolate])
    x = Nx.stack([step.x_old, step.x_new_rk_interpolate]) |> Nx.transpose()

    x_out = interpolate_fn.(t, x, step.k_vals, tadd)
    x_out |> Utils.columns_as_list(0, tadd_length - 1)
  end

  defp call_output_fn(step, output_fn) when is_nil(output_fn) do
    step
  end

  defp call_output_fn(step, output_fn) do
    result = output_fn.(Enum.reverse(step.t_new_chunk), Enum.reverse(step.x_new_chunk))
    %{step | terminal_output: result}
  end

  defp call_event_fn(step, event_fn, _interpolate_fn, _opts) when is_nil(event_fn) do
    step
  end

  defp call_event_fn(step, event_fn, interpolate_fn, opts) do
    # Pass opts to event_fn?
    result = event_fn.(step.t_new, step.x_new)

    case result.status do
      :continue ->
        step

      :halt ->
        new_step = compute_new_event_fn_step(event_fn, step, interpolate_fn, opts)

        %{
          step
          | terminal_event: :halt,
            x_new: new_step.x_new,
            t_new: new_step.t_new
        }
    end
  end

  defp compute_new_event_fn_step(event_fn, step, interpolate_fn, opts) do
    zero_fn = fn t ->
      x = interpolate_one_point(t, step, interpolate_fn)
      event_fn.(t, x) |> Map.get(:value) |> Nx.to_number()
    end

    root = NonlinearEqnRoot.find_zero(zero_fn, [Nx.to_number(step.t_old), Nx.to_number(step.t_new)], opts)
    x_new = interpolate_one_point(root.x, step, interpolate_fn)
    %ComputedStep{t_new: root.x, x_new: x_new, k_vals: step.k_vals, options_comp: step.options_comp}
  end
end
