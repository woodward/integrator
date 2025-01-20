defmodule Integrator.AdaptiveStepsize.InternalComputations do
  @moduledoc """
  Internal computations for `AdaptiveStepsize`
  """

  import Nx.Defn

  alias Integrator.ExternalFnAdapter
  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.NxOptions
  alias Integrator.Point
  alias Integrator.RungeKutta
  alias Integrator.RungeKutta.Step
  alias Integrator.NonLinearEqnRoot
  alias Integrator.Utils

  import Integrator.Utils, only: [timestamp_μs: 0]

  # Base zero_tolerance on precision?
  @zero_tolerance 1.0e-07

  @spec integrate_via_nx_while_loop(IntegrationStep.t(), Nx.t(), NxOptions.t()) :: IntegrationStep.t()
  defn integrate_via_nx_while_loop(starting_step, t_end, options) do
    {updated_step, _t_end, _options} =
      while {step = starting_step, t_end, options}, continue_stepping?(step, t_end) do
        compute_integration_step(step, t_end, options)
      end

    updated_step
  end

  @spec integrate_via_elixir_recursion(IntegrationStep.t(), Nx.t(), NxOptions.t()) :: IntegrationStep.t()
  deftransform integrate_via_elixir_recursion(step, t_end, options) do
    if continue_stepping?(step, t_end) == true_nx() do
      {step, t_end, options} = compute_integration_step(step, t_end, options)
      sleep_time_ms = compute_sleep_time(step, options)

      if sleep_time_ms && sleep_time_ms > 0 do
        Process.sleep(sleep_time_ms)
        %{step | step_timestamp_μs: timestamp_μs()}
      else
        step
      end
      |> integrate_via_elixir_recursion(t_end, options)
    else
      step
    end
  end

  @spec compute_integration_step(IntegrationStep.t(), Nx.t(), NxOptions.t()) :: {IntegrationStep.t(), Nx.t(), NxOptions.t()}
  defn compute_integration_step(step, t_end, options) do
    rk_step = RungeKutta.Step.compute_step(step.rk_step, step.dt_new, step.stepper_fn, step.ode_fn, options)
    step = step |> increment_compute_counter()

    step =
      if error_less_than_one?(rk_step) do
        step
        |> reset_error_count_to_zero()
        |> increment_counters()
        |> merge_rk_step_into_integration_step(rk_step)
        |> call_event_fn(options)
        |> interpolate_output_points(options)
      else
        step
        |> bump_error_count(options.max_number_of_errors)
      end

    dt_new =
      compute_next_timestep(step.dt_new, rk_step.error_estimate, options.order, step.t_current, t_end, options)

    {%{step | dt_new: dt_new}, t_end, options}
  end

  # Printing example - paste in where necessary:

  # {step, _} =
  #   hook({step, options}, fn {s, opts} ->
  #     IO.inspect(s.rk_step.t_new)
  #     IO.inspect(s.rk_step.x_new)
  #     {s, opts}
  #   end)

  # Another example with an output function:

  # step =
  #   if dt_last == Nx.f64(0.6746869564907434) do
  #     {step, _} =
  #       hook({step, options.output_fn_adapter}, fn {s, adapter} ->
  #         {t, x} = s.output_t_and_x_multi
  #         point = %Point{t: t, x: x}
  #         adapter.external_fn.(point)
  #         IO.inspect(Nx.to_number(s.dt_new), label: "step.dt_new - after")
  #         {s, adapter}
  #       end)

  #     step
  #   else
  #     step
  #   end

  @spec error_less_than_one?(IntegrationStep.t()) :: Nx.t()
  defn error_less_than_one?(rk_step) do
    rk_step.error_estimate < 1.0
  end

  @spec reset_error_count_to_zero(IntegrationStep.t()) :: IntegrationStep.t()
  defn reset_error_count_to_zero(step) do
    %{step | error_count: 0}
  end

  @spec increment_counters(IntegrationStep.t()) :: IntegrationStep.t()
  defn increment_counters(step) do
    %{
      step
      | count_loop__increment_step: step.count_loop__increment_step + 1,
        i_step: step.i_step + 1,
        fixed_output_t_within_step?: Nx.u8(0)
    }
  end

  @spec merge_rk_step_into_integration_step(IntegrationStep.t(), RungeKutta.Step.t()) :: IntegrationStep.t()
  defn merge_rk_step_into_integration_step(step, rk_step) do
    %{
      step
      | rk_step: rk_step,
        t_current: rk_step.t_new,
        x_current: rk_step.x_new
    }
  end

  @spec interpolate_output_points(IntegrationStep.t(), NxOptions.t()) :: IntegrationStep.t()
  defn interpolate_output_points(step, options) do
    cond do
      options.fixed_output_times? ->
        step |> interpolate_output_fixed_times(options)

      options.refine == 1 ->
        step |> output_single_point(options.output_fn_adapter, step.t_current, step.x_current)

      true ->
        # Note that step.t_current here is equal to step.rk_step.t_new (if no event fn has triggered
        # during this integration step) OR the time at which the event fn triggered (if an event did in
        # fact happen during this step):
        {t_add, x_out} = Step.interpolate_multiple_points(step.interpolate_fn, step.t_current, step.rk_step, options)
        step |> output_multiple_points(options.output_fn_adapter, t_add, x_out)
    end
  end

  @spec interpolate_output_fixed_times(IntegrationStep.t(), NxOptions.t()) :: IntegrationStep.t()
  defn interpolate_output_fixed_times(step, options) do
    {step, _} =
      while({step, options}, fixed_point_within_this_step?(step.fixed_output_t_next, step.rk_step.t_new)) do
        t_next_fixed = step.fixed_output_t_next
        x_out = Step.interpolate_single_specified_point(step.interpolate_fn, step.rk_step, t_next_fixed)
        step = step |> output_single_point(options.output_fn_adapter, t_next_fixed, x_out)
        step = %{step | fixed_output_t_next: t_next_fixed + options.fixed_output_step}
        {step, options}
      end

    step
  end

  defn true_nx, do: Nx.u8(1)
  defn false_nx, do: Nx.u8(0)

  @spec interpolate_fixed_points(IntegrationStep.t(), NxOptions.t()) :: IntegrationStep.t()
  defn interpolate_fixed_points(step, options) do
    fixed_output_t_next = step.fixed_output_t_next

    if fixed_point_within_this_step?(fixed_output_t_next, step.rk_step.t_new) do
      x_out = Step.interpolate_single_specified_point(step.interpolate_fn, step.rk_step, fixed_output_t_next)

      %{
        step
        | fixed_output_t_within_step?: true_nx(),
          output_t_and_x_single: {fixed_output_t_next, x_out},
          fixed_output_t_next: fixed_output_t_next + options.fixed_output_step
      }
    else
      %{step | fixed_output_t_within_step?: false_nx()}
    end
  end

  @spec fixed_point_within_this_step?(Nx.t(), Nx.t()) :: Nx.t()
  defnp fixed_point_within_this_step?(fixed_time, t_new) do
    fixed_time < t_new or Nx.abs(fixed_time - t_new) < @zero_tolerance
  end

  @spec output_single_point(IntegrationStep.t(), ExternalFnAdapter.t(), Nx.t(), Nx.t()) :: IntegrationStep.t()
  defn output_single_point(step, output_fn_adapter, t, x) do
    {step, _output_fn_adapter, _, _} =
      hook({step, output_fn_adapter, t, x}, fn {s, adapter, t_out, x_out} ->
        adapter.external_fn.(%Point{t: t_out, x: x_out})
        {s, adapter, t_out, x_out}
      end)

    step
  end

  @spec output_multiple_points(IntegrationStep.t(), ExternalFnAdapter.t(), Nx.t(), Nx.t()) :: IntegrationStep.t()
  defn output_multiple_points(step, output_fn_adapter, t_chunk, x_chunk) do
    {step, _, _, _} =
      hook({step, output_fn_adapter, t_chunk, x_chunk}, fn {s, adapter, t, x} ->
        t_list = Utils.vector_as_list(t)
        x_list = Utils.columns_as_list(x, 0)

        points =
          Enum.zip(t_list, x_list)
          |> Enum.map(fn {t_i, x_i} ->
            %Point{t: t_i, x: x_i}
          end)

        adapter.external_fn.(points)
        {s, adapter, t, x}
      end)

    step
  end

  @spec compute_sleep_time(IntegrationStep.t(), NxOptions.t(), pos_integer()) :: pos_integer() | nil
  deftransform compute_sleep_time(step, options, time_now_μs \\ :os.system_time(:microsecond)) do
    if Nx.to_number(step.error_count) > 0 or options.speed == Nx.Constants.infinity(options.type) do
      nil
    else
      t_new = Nx.to_number(step.t_current)
      t_old = Nx.to_number(step.rk_step.t_old)
      speed = Nx.to_number(options.speed)
      step_timestamp_μs = Nx.to_number(step.step_timestamp_μs)

      desired_time_interval_ms = (t_new - t_old) * 1000.0 / speed
      elapsed_time_ms = (time_now_μs - step_timestamp_μs) / 1000.0
      trunc(desired_time_interval_ms - elapsed_time_ms)
    end
  end

  @spec continue :: Nx.t()
  defn continue, do: Nx.u8(1)

  @spec halt :: Nx.t()
  defn halt, do: Nx.u8(0)

  @spec call_event_fn(IntegrationStep.t(), NxOptions.t()) :: IntegrationStep.t()
  defn call_event_fn(step, options) do
    event_fn_result = options.event_fn_adapter.external_fn.(step.t_current, step.x_current)

    if event_fn_result == continue() do
      step
    else
      {t_at_event_fn_zero, status_non_linear_eqn_root} = find_event_fn_t_zero(step, options)
      x_at_event_fn_zero = Step.interpolate_single_specified_point(step.interpolate_fn, step.rk_step, t_at_event_fn_zero)

      %{
        step
        | terminal_event: halt(),
          t_current: t_at_event_fn_zero,
          x_current: x_at_event_fn_zero,
          status_non_linear_eqn_root: status_non_linear_eqn_root
      }
    end
  end

  defn zero_fn(t_add, zero_fn_args) do
    [interpolate_fn, rk_step] = zero_fn_args
    x = Step.interpolate_single_specified_point(interpolate_fn, rk_step, t_add)
    x[0]
  end

  deftransform find_zero_fn(step) do
    fn a, b ->
      zero_fn(a, [step.interpolate_fn | b])
    end
  end

  defn find_event_fn_t_zero(step, options) do
    rk_step = step.rk_step
    t_old = rk_step.t_old
    t_new = rk_step.t_new

    zero_fn = find_zero_fn(step)
    zero_fn_args = [step.rk_step]

    root =
      NonLinearEqnRoot.find_zero_nx(
        zero_fn,
        t_old,
        t_new,
        zero_fn_args,
        options.non_linear_eqn_root_nx_options
      )

    {root.x, root.status}

    # An example root which satisfies the event_fn test:
    # Nx.f64(2.161317515510217)
  end

  @spec continue_stepping?(IntegrationStep.t(), Nx.t()) :: Nx.t()
  defnp continue_stepping?(step, t_end) do
    # Also check the step's status here

    #                                        if    close to end time                                  or past end time
    #  step.t_current > t_end do
    if step.terminal_event == halt() or Nx.abs(step.t_current - t_end) <= @zero_tolerance or
         step.t_current > t_end or step.status_integration > Nx.u8(1) do
      halt()
    else
      continue()
    end
  end

  @spec increment_compute_counter(IntegrationStep.t()) :: IntegrationStep.t()
  defnp increment_compute_counter(step) do
    %{step | count_cycles__compute_step: step.count_cycles__compute_step + 1}
  end

  @spec factor(Nx.t(), Nx.Type.t()) :: Nx.t()
  # defnp factor(order, nx_type), do: if(order == 3, do: @factor_order3, else: @factor_order5)
  defnp factor(order, nx_type) do
    Nx.tensor(0.38, type: nx_type) ** exponent(order, nx_type)
  end

  @spec exponent(Nx.t(), Nx.Type.t()) :: Nx.t()
  defnp exponent(order, nx_type), do: Nx.tensor(1, type: nx_type) / (Nx.tensor(1, type: nx_type) + order)

  @stepsize_factor_min 0.8
  @stepsize_factor_max 1.5

  # Formula taken from Hairer
  @spec compute_next_timestep(Nx.t(), Nx.t(), integer(), Nx.t(), Nx.t(), NxOptions.t()) :: Nx.t()
  defn compute_next_timestep(dt, error, order, t_old, t_end, options) do
    type = options.type

    # Avoid divisions by zero:
    # error = error + Nx.Constants.epsilon(type)
    error = error + Nx.Constants.epsilon(type)

    # Octave:
    #   dt *= min (facmax, max (facmin, fac * (1 / err)^(1 / (order + 1))));

    one = Nx.tensor(1.0, type: type)
    foo = factor(order, type) * (one / error) ** exponent(order, type)
    dt = dt * min(Nx.tensor(@stepsize_factor_max, type: type), max(Nx.tensor(@stepsize_factor_min, type: type), foo))
    dt = min(Nx.abs(dt), options.max_step)

    # Make sure we don't go past t_end:
    min(Nx.abs(dt), Nx.abs(t_end - t_old))
  end

  @spec bump_error_count(IntegrationStep.t(), Nx.t() | integer()) :: IntegrationStep.t()
  defnp bump_error_count(step, max_number_of_errors) do
    step = %{step | error_count: step.error_count + 1}

    if step.error_count > max_number_of_errors do
      %{step | status_integration: IntegrationStep.max_errors_exceeded()}
    else
      step
    end
  end

  @spec record_elapsed_time(IntegrationStep.t()) :: IntegrationStep.t()
  defn record_elapsed_time(step) do
    %{step | elapsed_time_μs: timestamp_μs() - step.start_timestamp_μs}
  end
end
