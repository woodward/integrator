defmodule Integrator.AdaptiveStepsize do
  @moduledoc """
  Integrates a set of ODEs with an adaptive timestep
  """
  import Nx.Defn

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
          dt: Nx.t() | nil,
          k_vals: Nx.t() | nil,
          #
          options_comp: Nx.t() | nil,
          fixed_times: [Nx.t()] | nil,
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
          ode_t: [Nx.t()],
          ode_x: [Nx.t()],
          #
          # The output of the integration, including the interpolated points:
          output_t: [Nx.t()],
          output_x: [Nx.t()],
          #
          # The last chunk of points; will include the computed point plus the interpolated points (if
          # interpolation is enabled) or just the computed point (if interpolation is disabled):
          t_new_chunk: [Nx.t()],
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

  # Base zero_tolerance on precision?
  @zero_tolerance 1.0e-07

  @nx_true Nx.tensor(1, type: :u8)

  @default_opts [
    max_number_of_errors: 5_000,
    max_step: 2.0,
    refine: 4,
    store_results?: true
  ]

  @doc """
  Integrates a set of ODEs. Originally adapted from
  [integrate_adaptive.m](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/integrate_adaptive.m)
  in Octave.

  See [Wikipedia](https://en.wikipedia.org/wiki/Adaptive_stepsize)
  """
  @spec integrate(
          stepper_fn :: fun(),
          interpolate_fn :: fun(),
          ode_fn :: fun(),
          t_start :: Nx.t(),
          t_end :: Nx.t(),
          fixed_times :: [Nx.t()] | nil,
          initial_tstep :: Nx.t(),
          x0 :: Nx.t(),
          order :: integer(),
          opts :: Keyword.t()
        ) :: t()
  def integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, fixed_times, initial_tstep, x0, order, opts \\ []) do
    opts = @default_opts |> Keyword.merge(Utils.default_opts()) |> Keyword.merge(opts)
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
    |> store_first_point(t_start, x0, opts[:store_results?])
    |> step_forward(Nx.to_number(t_start), Nx.to_number(t_end), :continue, stepper_fn, interpolate_fn, ode_fn, order, opts)
    |> reverse_results()
  end

  # ===========================================================================
  # Private functions below here:

  @spec drop_first_point([Nx.t()] | nil) :: [Nx.t()] | nil
  defp drop_first_point(nil), do: nil

  defp drop_first_point(fixed_times) do
    [_drop_first_point | rest_of_fixed_times] = fixed_times
    rest_of_fixed_times
  end

  @spec store_first_point(t(), Nx.t(), Nx.t(), boolean()) :: t()
  defp store_first_point(step, t_start, x0, true = _store_results?) do
    %{step | output_t: [t_start], output_x: [x0], ode_t: [t_start], ode_x: [x0]}
  end

  defp store_first_point(step, _t_start, _x0, _store_results?), do: step

  @spec initial_empty_k_vals(integer(), Nx.t()) :: Nx.t()
  defp initial_empty_k_vals(order, x) do
    # Figure out the correct way to do this!  Does k_length depend on the order of the Runge Kutta method?
    k_length = order + 2

    {length_x} = Nx.shape(x)
    Nx.broadcast(0.0, {length_x, k_length})
  end

  @spec step_forward(
          step :: t(),
          t_old :: float(),
          t_end :: float(),
          status :: integration_status(),
          stepper_fn :: fun(),
          interpolate_fn :: fun(),
          ode_fn :: fun(),
          order :: integer(),
          opts :: Keyword.t()
        ) :: t()
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
        |> store_resuts(opts[:store_results?])
        |> call_output_fn(opts[:output_fn])
      else
        bump_error_count(step, opts)
      end

    dt = compute_next_timestep(step.dt, error_est, order, step.t_new, t_end, opts)
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

  @spec t_next(t(), Nx.t()) :: float()
  defp t_next(%{error_count: error_count} = step, dt) when error_count > 0 do
    # Update this into step somehow???
    Nx.add(step.t_old, dt) |> Nx.to_number()
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
  @spec compute_next_timestep(Nx.t(), Nx.t(), integer(), Nx.t(), Nx.t(), Keyword.t()) :: Nx.t()
  defnp compute_next_timestep(dt, error, order, t_old, t_end, opts) do
    # # Avoid divisions by zero:
    error = error + epsilon(opts[:type])

    # factor should be cached somehow; perhaps passed in in the options?
    factor = 0.38 ** (1.0 / (order + 1))

    foo = factor * (1 / error ** (1 / (order + 1)))

    dt = dt * min(@stepsize_factor_max, max(@stepsize_factor_min, foo))
    dt = min(Nx.abs(dt), opts[:max_step])

    # # ## Make sure we don't go past t_end:
    min(Nx.abs(dt), Nx.abs(t_end - t_old))
  end

  # What should the typespec be for a deftransformp?
  @spec epsilon(any()) :: float()
  deftransformp epsilon(type) do
    Utils.epsilon(type)
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
  defp store_resuts(step, false = _store_results?) do
    step
  end

  defp store_resuts(step, true = _store_results?) do
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
    k_vals_old = step.k_vals
    dt = step.dt

    {t_next, x_next, k_vals, options_comp, error} =
      compute_step_nx(stepper_fn, ode_fn, t_old, x_old, k_vals_old, options_comp_old, dt, opts)

    {%ComputedStep{
       t_new: t_next,
       x_new: x_next,
       k_vals: k_vals,
       options_comp: options_comp
     }, error}
  end

  @spec compute_step_nx(
          stepper_fn :: fun(),
          ode_fn :: fun(),
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
    {_t_new, options_comp} = Utils.kahan_sum(t_old, options_comp_old, dt)
    {t_next, x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals_old)
    error = Utils.abs_rel_norm(x_next, x_old, x_est, opts[:abs_tol], opts[:rel_tol], norm_control: opts[:norm_control])
    {t_next, x_next, k_vals, options_comp, error}
  end

  @spec add_fixed_point(t(), fun()) :: t()
  defp add_fixed_point(%{fixed_times: []} = step, _interpolate_fn) do
    step
  end

  defp add_fixed_point(step, interpolate_fn) do
    [fixed_time | remaining_fixed_times] = step.fixed_times

    if add_fixed_point?(fixed_time, step.t_new) == @nx_true do
      x_at_fixed_time = interpolate_one_point(fixed_time, step, interpolate_fn)

      step = %{
        step
        | t_new_chunk: [fixed_time | step.t_new_chunk],
          x_new_chunk: [x_at_fixed_time | step.x_new_chunk],
          fixed_times: remaining_fixed_times
      }

      add_fixed_point(step, interpolate_fn)
    else
      step
    end
  end

  # @spec add_fixed_point?(Nx.t(), Nx.t()) :: Nx.t()
  defnp add_fixed_point?(fixed_time, t_new) do
    fixed_time < t_new or Nx.abs(fixed_time - t_new) < @zero_tolerance
  end

  @spec interpolate(t(), fun(), refine_strategy()) :: t()
  defp interpolate(step, interpolate_fn, refine) when refine == :fixed_times do
    add_fixed_point(%{step | t_new_chunk: [], x_new_chunk: []}, interpolate_fn)
  end

  defp interpolate(step, _interpolate_fn, refine) when refine == 1 do
    %{step | t_new_chunk: [step.t_new], x_new_chunk: [step.x_new]}
  end

  defp interpolate(step, interpolate_fn, refine) when refine > 1 do
    tadd = Nx.linspace(step.t_old, step.t_new, n: refine + 1, type: Nx.type(step.x_old))
    # Get rid of the first element (tadd[0]) via this slice:
    tadd = Nx.slice_along_axis(tadd, 1, refine, axis: 0)

    x_out_as_cols = do_interpolation(step, interpolate_fn, tadd) |> Enum.reverse()
    t_new_chunk = tadd |> Utils.vector_as_list() |> Enum.reverse()
    %{step | x_new_chunk: x_out_as_cols, t_new_chunk: t_new_chunk}
  end

  @spec interpolate_one_point(Nx.t(), t(), fun()) :: Nx.t()
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
end
