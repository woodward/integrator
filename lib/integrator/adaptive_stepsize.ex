defmodule Integrator.AdaptiveStepsize do
  @moduledoc """
  Integrates a set of ODEs with an adaptive timestep
  """

  alias Integrator.{MaxErrorsExceededError, NonlinearEqnRoot, Utils}

  defmodule ComputedStep do
    @moduledoc false

    @type t :: %__MODULE__{
            t_new: Nx.t(),
            x_new: Nx.t(),
            k_vals: Nx.t(),
            options_comp: Nx.t()
          }

    defstruct [
      :t_new,
      :x_new,
      :k_vals,
      :options_comp
    ]
  end

  @type t :: %__MODULE__{
          t_old: Nx.t() | nil,
          x_old: Nx.t() | nil,
          #
          t_new: Nx.t() | nil,
          x_new: Nx.t() | nil,
          #
          t_new_rk_interpolate: Nx.t() | nil,
          x_new_rk_interpolate: Nx.t() | nil,
          #
          dt: float(),
          k_vals: Nx.t() | nil,
          #
          options_comp: Nx.t() | nil,
          fixed_times: [float()] | nil,
          #
          count_loop__increment_step: integer(),
          count_cycles__compute_step: integer(),
          #
          # ireject in Octave:
          error_count: integer(),
          i_step: integer(),
          #
          terminal_event: integration_status(),
          terminal_output: integration_status(),
          #
          # The output of the integration:
          ode_t: [float()],
          ode_x: [Nx.t()],
          #
          # The output of the integration, including the interpolated points:
          output_t: [float()],
          output_x: [Nx.t()],
          #
          # The last chunk of points; will include the computed point plus the interpolated points (if
          # interpolation is enabled) or just the computed point (if interpolation is disabled):
          t_new_chunk: [float()],
          x_new_chunk: [Nx.t()]
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
    fixed_times: nil,
    #
    count_loop__increment_step: 0,
    count_cycles__compute_step: 0,
    #
    # ireject in Octave:
    error_count: 0,
    i_step: 0,
    #
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

  @type integration_status :: :halt | :continue
  @type refine_strategy :: integer() | :fixed_times

  @stepsize_factor_min 0.8
  @stepsize_factor_max 1.5

  @default_refine 4
  @default_max_number_of_errors 5_000
  @default_max_step 2.0
  @default_store_resuts true

  @zero_tolerance 1.0e-07

  # Switch to using Utils.epsilon/1
  @epsilon 2.2204e-16

  @nx_true Nx.tensor(1, type: :u8)

  @doc """
  Integrates a set of ODEs. Originally adapted from
  [integrate_adaptive.m](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/integrate_adaptive.m)
  in Octave.

  See [Wikipedia](https://en.wikipedia.org/wiki/Adaptive_stepsize)
  """
  @spec integrate(fun(), fun(), fun(), float(), float(), Nx.t() | nil, float, Nx.t(), integer(), Keyword.t()) :: t()
  def integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, fixed_times, initial_tstep, x0, order, opts \\ []) do
    opts = default_opts() |> Keyword.merge(Utils.default_opts()) |> Keyword.merge(opts)
    fixed_times = fixed_times |> drop_first_point()

    # Broadcast the starting conditions (t_start & x0) as the first output point (if there is an output function):
    if fun = opts[:output_fn], do: fun.([t_start], [x0])

    # If there are fixed output times, then refine can no longer be an integer value (such as 1 or 4):
    opts = if fixed_times, do: Keyword.merge(opts, refine: :fixed_times), else: opts

    %__MODULE__{
      t_new: t_start,
      x_new: x0,
      dt: initial_tstep,
      k_vals: initial_empty_k_vals(order, x0),
      fixed_times: fixed_times
    }
    |> store_first_point(t_start, x0, opts[:store_resuts?])
    |> step_forward(t_start, t_end, :continue, stepper_fn, interpolate_fn, ode_fn, order, opts)
    |> reverse_results()
  end

  # ===========================================================================
  # Private functions below here:

  @spec drop_first_point([float()] | nil) :: [float()] | nil
  defp drop_first_point(nil), do: nil

  defp drop_first_point(fixed_times) do
    [_drop_first_point | rest_of_fixed_times] = fixed_times
    rest_of_fixed_times
  end

  @spec store_first_point(t(), float(), Nx.t(), boolean()) :: t()
  defp store_first_point(step, t_start, x0, true = _store_resuts?) do
    %{step | output_t: [t_start], output_x: [x0], ode_t: [t_start], ode_x: [x0]}
  end

  defp store_first_point(step, _t_start, _x0, _store_resuts?), do: step

  @spec initial_empty_k_vals(integer(), Nx.t()) :: Nx.t()
  defp initial_empty_k_vals(order, x) do
    # Figure out the correct way to do this!  Does k_length depend on the order of the Runge Kutta method?
    k_length = order + 2

    {length_x} = Nx.shape(x)
    Nx.broadcast(0.0, {length_x, k_length})
  end

  @spec step_forward(t(), float(), float(), integration_status(), fun(), fun(), fun(), integer(), Keyword.t()) :: t()
  defp step_forward(step, t_old, t_end, _status, _stepper_fn, _interpolate_fn, _ode_fn, _order, _opts)
       when abs(t_old - t_end) < @zero_tolerance or t_old > t_end do
    step
  end

  defp step_forward(step, _t_old, _t_end, status, _stepper_fn, _interpolate_fn, _ode_fn, _order, _opts)
       when status == :halt do
    step
  end

  defp step_forward(step, _t_old, t_end, _status, stepper_fn, interpolate_fn, ode_fn, order, opts) do
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

  @spec less_than_one?(Nx.t()) :: boolean()
  defp less_than_one?(error_est), do: Nx.less(error_est, 1.0) == @nx_true

  @spec halt?(t()) :: integration_status()
  defp halt?(%{terminal_event: :halt} = _step), do: :halt
  defp halt?(%{terminal_output: :halt} = _step), do: :halt
  defp halt?(_step), do: :continue

  @spec bump_error_count(t(), Keyword.t()) :: t()
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

  @spec t_next(t(), float()) :: float()
  defp t_next(%{error_count: error_count} = step, dt) when error_count > 0 do
    # Update this into step somehow???
    Nx.to_number(step.t_old) + dt
  end

  defp t_next(%{error_count: error_count} = step, _dt) when error_count == 0 do
    Nx.to_number(step.t_new)
  end

  @spec reverse_results(t()) :: t()
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
  @spec compute_next_timestep(float(), float(), integer(), float(), float(), Keyword.t()) :: float()
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

  @spec increment_and_reset_counters(t()) :: t()
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

  @spec merge_new_step(t(), ComputedStep.t()) :: t()
  defp merge_new_step(step, computed_step) do
    %{
      step
      | x_old: step.x_new,
        t_old: step.t_new,
        #
        x_new: computed_step.x_new,
        t_new: computed_step.t_new,
        #
        x_new_rk_interpolate: computed_step.x_new,
        t_new_rk_interpolate: computed_step.t_new,
        #
        k_vals: computed_step.k_vals,
        options_comp: computed_step.options_comp
    }
  end

  @spec store_resuts(t(), boolean()) :: t()
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

  @spec increment_compute_counter(t()) :: t()
  defp increment_compute_counter(step) do
    %{step | count_cycles__compute_step: step.count_cycles__compute_step + 1}
  end

  @spec compute_step(t(), fun(), fun(), Keyword.t()) :: {ComputedStep.t(), Nx.t()}
  defp compute_step(step, stepper_fn, ode_fn, opts) do
    x_old = step.x_new
    t_old = step.t_new
    options_comp_old = step.options_comp
    k_vals = step.k_vals
    dt = step.dt

    {_t_new, options_comp} = Utils.kahan_sum(t_old, options_comp_old, dt)
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

  @spec add_fixed_point(t(), [float()], [Nx.t()], fun()) :: {t(), [float()], [Nx.t()]}
  defp add_fixed_point(%{fixed_times: []} = step, new_t_chunk, new_x_chunk, _interpolate_fn) do
    step = %{step | t_new_chunk: [Nx.to_number(step.t_new)], x_new_chunk: [step.x_new]}
    {step, new_t_chunk, new_x_chunk}
  end

  defp add_fixed_point(%{fixed_times: fixed_times} = step, new_t_chunk, new_x_chunk, interpolate_fn) do
    [new_time | remaining_times] = fixed_times
    t_new = Nx.to_number(step.t_new)

    if new_time < t_new || abs(new_time - t_new) < @zero_tolerance do
      step = %{step | fixed_times: remaining_times}
      x_new = interpolate_one_point(new_time, step, interpolate_fn)
      add_fixed_point(step, [new_time | new_t_chunk], [x_new | new_x_chunk], interpolate_fn)
    else
      {step, new_t_chunk, new_x_chunk}
    end
  end

  @spec interpolate(t(), fun(), refine_strategy()) :: t()
  defp interpolate(step, interpolate_fn, refine) when refine == :fixed_times do
    {step, new_t_chunk, new_x_chunk} = add_fixed_point(step, [], [], interpolate_fn)
    %{step | t_new_chunk: new_t_chunk, x_new_chunk: new_x_chunk}
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

  @spec interpolate_one_point(float(), t(), fun()) :: Nx.t()
  defp interpolate_one_point(t_new, step, interpolate_fn) do
    do_interpolation(step, interpolate_fn, Nx.tensor(t_new)) |> List.first()
  end

  @spec do_interpolation(t(), fun(), Nx.t()) :: [Nx.t()]
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

  @spec call_output_fn(t(), fun()) :: t()
  defp call_output_fn(step, output_fn) when is_nil(output_fn) do
    step
  end

  defp call_output_fn(step, output_fn) do
    result = output_fn.(Enum.reverse(step.t_new_chunk), Enum.reverse(step.x_new_chunk))
    %{step | terminal_output: result}
  end

  @spec call_event_fn(t(), fun(), fun(), Keyword.t()) :: t()
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
        new_step = step |> compute_new_event_fn_step(event_fn, interpolate_fn, opts)

        %{
          step
          | terminal_event: :halt,
            x_new: new_step.x_new,
            t_new: new_step.t_new
        }
    end
  end

  @spec compute_new_event_fn_step(t(), fun(), fun(), Keyword.t()) :: ComputedStep.t()
  defp compute_new_event_fn_step(step, event_fn, interpolate_fn, opts) do
    zero_fn = fn t ->
      x = interpolate_one_point(t, step, interpolate_fn)
      event_fn.(t, x) |> Map.get(:value) |> Nx.to_number()
    end

    root = NonlinearEqnRoot.find_zero(zero_fn, [Nx.to_number(step.t_old), Nx.to_number(step.t_new)], opts)
    x_new = interpolate_one_point(root.x, step, interpolate_fn)
    %ComputedStep{t_new: root.x, x_new: x_new, k_vals: step.k_vals, options_comp: step.options_comp}
  end

  @spec default_opts() :: Keyword.t()
  defp default_opts() do
    [
      epsilon: @epsilon,
      max_number_of_errors: @default_max_number_of_errors,
      max_step: @default_max_step,
      refine: @default_refine,
      store_resuts?: @default_store_resuts
    ]
  end
end