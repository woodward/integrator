defmodule Integrator.AdaptiveStepsize do
  @moduledoc """
  Integrates a set of ODEs with an adaptive timestep.
  """
  import Nx.Defn
  import Nx, only: :sigils

  alias Integrator.{MaxErrorsExceededError, NonLinearEqnRoot, Utils}
  alias Integrator.AdaptiveStepsize.ArgPrecisionError

  defmodule ComputedStep do
    @moduledoc """
    The results of the computation of an individual Runge-Kutta step
    """

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
          # t & x used for Runge-Kutta interpolations:
          t_new_rk_interpolate: Nx.t() | nil,
          x_new_rk_interpolate: Nx.t() | nil,
          #
          dt: Nx.t() | nil,
          k_vals: Nx.t() | nil,
          nx_type: Nx.Type.t(),
          #
          options_comp: Nx.t() | nil,
          #
          # Fixed output times; e.g., integration output computed at times at [0.1, 0.2, 0.3, ...] via interpolation:
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
          # The output of the Runge-Kutta integration:
          ode_t: [Nx.t()],
          ode_x: [Nx.t()],
          #
          # The output of the integration, plus the interpolated points:
          output_t: [Nx.t()],
          output_x: [Nx.t()],
          #
          # The last chunk of points for this computed step; will include the computed point plus the
          # interpolated points (if # interpolation is enabled) or just the computed point (if interpolation is disabled):
          t_new_chunk: [Nx.t()],
          x_new_chunk: [Nx.t()],
          #
          timestamp_ms: integer() | nil,
          timestamp_start_ms: integer() | nil
        }
  defstruct [
    :t_old,
    :x_old,
    #
    :t_new,
    :x_new,
    #
    # t & x used for Runge-Kutta interpolations:
    :t_new_rk_interpolate,
    :x_new_rk_interpolate,
    #
    :dt,
    :k_vals,
    nx_type: :f32,
    #
    options_comp: 0.0,
    #
    # Fixed output times; e.g., integration output computed at times at [0.1, 0.2, 0.3, ...] via interpolation:
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
    # The output of the Runge-Kutta integration:
    ode_t: [],
    ode_x: [],
    #
    # The output of the integration, plus the interpolated points:
    output_t: [],
    output_x: [],
    #
    # The last chunk of points for this computed step; will include the computed point plus the
    # interpolated points (if # interpolation is enabled) or just the computed point (if interpolation is disabled):
    t_new_chunk: [],
    x_new_chunk: [],
    #
    timestamp_ms: nil,
    timestamp_start_ms: nil
  ]

  defmodule MaxErrorsExceededError do
    defexception message: "Too many errors",
                 error_count: 0,
                 max_number_of_errors: 0,
                 step: nil
  end

  defmodule ArgPrecisionError do
    defexception message: "argument precision error",
                 invalid_argument: nil,
                 argument_name: nil,
                 expected_precision: nil,
                 actual_precision: nil
  end

  @type integration_status :: :halt | :continue
  @type refine_strategy :: integer() | :fixed_times

  # Base zero_tolerance on precision?
  @zero_tolerance 1.0e-07

  @nx_true Nx.tensor(1, type: :u8)

  @default_opts [
    max_number_of_errors: 5_000,
    refine: 4,
    store_results?: true,
    speed: :no_delay
  ]

  @default_max_step %{f32: ~V[ 2.0 ]f32, f64: ~V[ 2.0 ]f64}

  @abs_rel_norm_opts %{
    f32: [abs_tol: ~V[ 1.0e-06 ]f32, rel_tol: ~V[ 1.0e-03 ]f32, norm_control: true],
    f64: [abs_tol: ~V[ 1.0e-06 ]f64, rel_tol: ~V[ 1.0e-03 ]f64, norm_control: true]
  }

  # :no_delay means to perform the integration as fast as possible
  # For float values, 1.0 means to integrate in real-time, 0.5 means half of real-time, 2.0 means twice as fast as real time, etc.
  @type speed :: :no_delay | float()

  @doc """
  Integrates a set of ODEs. Originally adapted from the Octave
  [integrate_adaptive.m](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/integrate_adaptive.m)

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
    nx_type = opts[:type]
    opts = (@default_opts ++ @abs_rel_norm_opts[nx_type] ++ [max_step: @default_max_step[nx_type]]) |> Keyword.merge(opts)
    fixed_times = fixed_times |> drop_first_point()

    # Broadcast the starting conditions (t_start & x0) as the first output point (if there is an output function):
    if fun = opts[:output_fn], do: fun.([t_start], [x0])

    opts =
      if fixed_times do
        # Spot-check the Nx type of the first time value in the list of fixed times:
        check_nx_type([fixed_times: hd(fixed_times)], nx_type)

        # If there are fixed output times, then refine can no longer be an integer value (such as 1 or 4):
        Keyword.merge(opts, refine: :fixed_times)
      else
        opts
      end

    check_nx_type(
      [
        t_start: t_start,
        t_end: t_end,
        x0: x0,
        initial_tstep: initial_tstep,
        abs_tol: opts[:abs_tol],
        rel_tol: opts[:rel_tol],
        max_step: opts[:max_step],
        ode_fn: ode_fn.(t_start, x0)
      ],
      nx_type
    )

    timestamp_now = timestamp_ms()

    %__MODULE__{
      t_new: t_start,
      x_new: x0,
      # t_old must be set on the initial struct in case there's an error when computing the first step (used in t_next/2)
      t_old: t_start,
      dt: initial_tstep,
      k_vals: initial_empty_k_vals(order, x0),
      fixed_times: fixed_times,
      nx_type: nx_type,
      options_comp: Nx.tensor(0.0, type: nx_type),
      timestamp_ms: timestamp_now,
      timestamp_start_ms: timestamp_now
    }
    |> store_first_point(t_start, x0, opts[:store_results?])
    |> step_forward(Nx.to_number(t_start), Nx.to_number(t_end), :continue, stepper_fn, interpolate_fn, ode_fn, order, opts)
    |> reverse_results()
    # Capture end timestamp:
    |> Map.put(:timestamp_ms, timestamp_ms())
  end

  @doc """
  Computes a good initial timestep for an ODE solver of order `order`
  using the algorithm described in the reference below.

  The input argument `ode_fn`, is the function describing the differential
  equations, `t0` is the initial time, and `x0` is the initial
  condition.  `abs_tol` and `rel_tol` are the absolute and relative
  tolerance on the ODE integration.

  Originally based on the Octave
  [`starting_stepsize.m`](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/starting_stepsize.m).

  Reference:

  E. Hairer, S.P. Norsett and G. Wanner,
  "Solving Ordinary Differential Equations I: Nonstiff Problems",
  Springer.
  """
  @spec starting_stepsize(
          order :: integer(),
          ode_fn :: fun(),
          t0 :: Nx.t(),
          x0 :: Nx.t(),
          abs_tol :: Nx.t(),
          rel_tol :: Nx.t(),
          opts :: Keyword.t()
        ) :: Nx.t()
  defn starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, opts \\ []) do
    nx_type = Nx.type(x0)
    # Compute norm of initial conditions
    x_zeros = zero_vector(x0)
    d0 = abs_rel_norm(x0, x0, x_zeros, abs_tol, rel_tol, opts)

    x = ode_fn.(t0, x0)

    d1 = abs_rel_norm(x, x, x_zeros, abs_tol, rel_tol, opts)

    h0 =
      if d0 < 1.0e-5 or d1 < 1.0e-5 do
        Nx.tensor(1.0e-6, type: nx_type)
      else
        Nx.tensor(0.01, type: nx_type) * (d0 / d1)
      end

    # Compute one step of Explicit-Euler
    x1 = x0 + h0 * x

    # Approximate the derivative norm
    xh = ode_fn.(t0 + h0, x1)

    xh_minus_x = xh - x
    d2 = Nx.tensor(1.0, type: nx_type) / h0 * abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, opts)

    one = Nx.tensor(1, type: nx_type)

    h1 =
      if max(d1, d2) <= 1.0e-15 do
        max(Nx.tensor(1.0e-06, type: nx_type), h0 * Nx.tensor(1.0e-03, type: nx_type))
      else
        Nx.pow(Nx.tensor(1.0e-02, type: nx_type) / max(d1, d2), one / (order + one))
      end

    min(Nx.tensor(100.0, type: nx_type) * h0, h1)
  end

  @doc """
  Gets the default values used by the absolute-relative norm; e.g., `abs_tol`, `rel_tol`, and
  `norm_control`
  """
  @spec abs_rel_norm_opts(atom) :: Keyword.t()
  def abs_rel_norm_opts(nx_type), do: @abs_rel_norm_opts[nx_type]

  @doc """
  Returns the total elapsed time for the integration (in milleseconds)
  """
  @spec elapsed_time_ms(t()) :: pos_integer()
  def elapsed_time_ms(step), do: step.timestamp_ms - step.timestamp_start_ms

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
    zero = Nx.tensor(0.0, type: Nx.type(x))
    Nx.broadcast(zero, {length_x, k_length})
  end

  #  The main integration loop
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
    step = %{step | dt: dt} |> delay_simulation(opts[:speed])

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

  # Results are stored on lists in reverse order for speed; reverse them before returning them to the user
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

  @one Nx.tensor(1.0, type: :f64)
  @three Nx.tensor(3.0, type: :f64)
  @five Nx.tensor(5.0, type: :f64)

  # exponent = one / (order + one)
  @exponent_order3 Nx.divide(@one, Nx.add(@three, @one))
  @exponent_order5 Nx.divide(@one, Nx.add(@five, @one))

  # factor = Nx.tensor(0.38, type: nx_type) ** exponent
  @zero_three_eight Nx.tensor(0.38, type: :f64)

  @factor_order3 Nx.pow(@zero_three_eight, @exponent_order3)
  @factor_order5 Nx.pow(@zero_three_eight, @exponent_order5)

  @stepsize_factor_min %{f32: Nx.tensor(0.8, type: :f32), f64: Nx.tensor(0.8, type: :f64)}
  @stepsize_factor_max %{f32: Nx.tensor(1.5, type: :f32), f64: Nx.tensor(1.5, type: :f64)}

  @spec exponent(Nx.t()) :: Nx.t()
  defnp exponent(order), do: if(order == 3, do: @exponent_order3, else: @exponent_order5)

  @spec factor(Nx.t()) :: Nx.t()
  defnp factor(order), do: if(order == 3, do: @factor_order3, else: @factor_order5)

  # Formula taken from Hairer
  @spec compute_next_timestep(Nx.t(), Nx.t(), integer(), Nx.t(), Nx.t(), Keyword.t()) :: Nx.t()
  defnp compute_next_timestep(dt, error, order, t_old, t_end, opts) do
    nx_type = opts[:type]

    # Avoid divisions by zero:
    error = error + epsilon(nx_type)

    # Octave:
    #   dt *= min (facmax, max (facmin, fac * (1 / err)^(1 / (order + 1))));

    one = Nx.tensor(1.0, type: nx_type)
    foo = factor(order) * (one / error) ** exponent(order)
    dt = dt * min(@stepsize_factor_max[nx_type], max(@stepsize_factor_min[nx_type], foo))
    dt = min(Nx.abs(dt), opts[:max_step])

    # Make sure we don't go past t_end:
    min(Nx.abs(dt), Nx.abs(t_end - t_old))
  end

  # What should the typespec be for a deftransformp?
  # @spec epsilon(atom()) :: float()
  @spec epsilon({:f, 32 | 64}) :: float()
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
    }
  end

  # Merges a newly-computed Runge-Kutta step into the AdaptiveStepsize struct
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

  # Inserts a delay for performing real-time simulations
  @spec delay_simulation(t(), speed()) :: t()
  defp delay_simulation(step, :no_delay), do: step

  defp delay_simulation(%{error_count: error_count} = step, _speed) when error_count > 0, do: step

  defp delay_simulation(step, speed) do
    t_new = Nx.to_number(step.t_new) * 1000
    t_now = timestamp_ms() - step.timestamp_start_ms
    t_diff = trunc((t_new - t_now) / speed)
    if t_diff > 0, do: Process.sleep(t_diff)
    %{step | timestamp_ms: timestamp_ms()}
  end

  # Computes the next Runge-Kutta step. Note that this function "wraps" the Nx functions which
  # perform the actual numerical computations
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

  # Computes the next Runge-Kutta step and the associated error
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
    {t_next, options_comp} = Utils.kahan_sum(t_old, options_comp_old, dt)
    {x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals_old, t_next)
    error = abs_rel_norm(x_next, x_old, x_est, opts[:abs_tol], opts[:rel_tol], norm_control: opts[:norm_control])
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
    do_interpolation(step, interpolate_fn, Nx.tensor(t_new, type: step.nx_type)) |> List.first()
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

  # Calls an output function (such as for plotting while the simulation is in progress)
  @spec call_output_fn(t(), fun()) :: t()
  defp call_output_fn(step, output_fn) when is_nil(output_fn) do
    step
  end

  defp call_output_fn(step, output_fn) do
    result = output_fn.(Enum.reverse(step.t_new_chunk), Enum.reverse(step.x_new_chunk))
    %{step | terminal_output: result}
  end

  # Calls an event function (e.g., checking to see if a bouncing ball has collided with a surface)
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

  # Hones in (via interpolation) on the exact point that the event function goes to zero
  @spec compute_new_event_fn_step(t(), fun(), fun(), Keyword.t()) :: ComputedStep.t()
  defp compute_new_event_fn_step(step, event_fn, interpolate_fn, opts) do
    zero_fn = fn t ->
      x = interpolate_one_point(t, step, interpolate_fn)
      event_fn.(t, x) |> Map.get(:value) |> Nx.to_number()
    end

    root = NonLinearEqnRoot.find_zero(zero_fn, [Nx.to_number(step.t_old), Nx.to_number(step.t_new)], opts)
    x_new = interpolate_one_point(root.x, step, interpolate_fn)
    %ComputedStep{t_new: Nx.tensor(root.x, type: opts[:type]), x_new: x_new, k_vals: step.k_vals, options_comp: step.options_comp}
  end

  # Originally based on
  # [Octave function AbsRelNorm](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/AbsRel_norm.m)

  # Options
  # * `:norm_control` - Control error relative to norm; i.e., control the error `e` at each step using the norm of the
  #   solution rather than its absolute value.  Defaults to true.

  # See [Matlab documentation](https://www.mathworks.com/help/matlab/ref/odeset.html#bu2m9z6-NormControl)
  # for a description of norm control.
  @spec abs_rel_norm(Nx.t(), Nx.t(), Nx.t(), float(), float(), Keyword.t()) :: Nx.t()
  defnp abs_rel_norm(t, t_old, x, abs_tolerance, rel_tolerance, opts \\ []) do
    if opts[:norm_control] do
      # Octave code
      # sc = max (AbsTol(:), RelTol * max (sqrt (sumsq (t)), sqrt (sumsq (t_old))));
      # retval = sqrt (sumsq ((t - x))) / sc;

      max_sq_t = Nx.max(sum_sq(t), sum_sq(t_old))
      sc = Nx.max(abs_tolerance, rel_tolerance * max_sq_t)
      sum_sq(t - x) / sc
    else
      # Octave code:
      # sc = max (AbsTol(:), RelTol .* max (abs (t), abs (t_old)));
      # retval = max (abs (t - x) ./ sc);

      sc = Nx.max(abs_tolerance, rel_tolerance * Nx.max(Nx.abs(t), Nx.abs(t_old)))
      (Nx.abs(t - x) / sc) |> Nx.reduce_max()
    end
  end

  # Sums the squares of a vector and then takes the square root (e.g., computes
  # the norm of a vector)
  @spec sum_sq(Nx.t()) :: Nx.t()
  defnp sum_sq(x) do
    (x * x) |> Nx.sum() |> Nx.sqrt()
  end

  # Creates a zero vector that has the length of `x`
  # Is there a better built-in Nx way of doing this?
  @spec zero_vector(Nx.t()) :: Nx.t()
  defnp zero_vector(x) do
    {length_of_x} = Nx.shape(x)
    zero = Nx.tensor(0.0, type: Nx.type(x))
    Nx.broadcast(zero, {length_of_x})
  end

  # Checks that the Nx types are in line with what is expected. This avoids args with mismatched types.
  @spec check_nx_type(Keyword.t(), Nx.Type.t()) :: atom()
  defp check_nx_type(args, expected_nx_type) do
    args
    |> Enum.each(fn {arg_name, arg_value} ->
      nx_type = Utils.nx_type_atom(arg_value)

      if nx_type != expected_nx_type do
        raise ArgPrecisionError,
          invalid_argument: arg_value,
          expected_precision: expected_nx_type,
          actual_precision: nx_type,
          argument_name: arg_name
      end
    end)

    :ok
  end

  # Returns a timestamp in milliseconds
  @spec timestamp_ms() :: pos_integer()
  defp timestamp_ms(), do: :os.system_time(:millisecond)
end
