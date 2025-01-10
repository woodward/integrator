defmodule Integrator.AdaptiveStepsize.InternalComputations do
  @moduledoc """
  Internal computations for `AdaptiveStepsize`
  """

  import Nx.Defn

  alias Integrator.ExternalFnAdapter
  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.MaxErrorsExceededError
  alias Integrator.AdaptiveStepsizeRefactor.NxOptions
  alias Integrator.Point
  alias Integrator.RungeKutta
  alias Integrator.RungeKutta.Step
  alias Integrator.Utils

  import Integrator.Utils, only: [timestamp_μs: 0]

  @spec integrate_step(IntegrationStep.t(), Nx.t(), NxOptions.t()) :: IntegrationStep.t()
  defn integrate_step(step_start, t_end, options) do
    {updated_step, _t_end, _options} =
      while {step = step_start, t_end, options}, continue_stepping?(step, t_end) do
        rk_step = RungeKutta.Step.compute_step(step.rk_step, step.dt_new, step.stepper_fn, step.ode_fn, options)
        step = step |> increment_compute_counter()

        step =
          if error_less_than_one?(rk_step) do
            step
            |> reset_error_count_to_zero()
            |> increment_counters()
            |> merge_rk_step_into_integration_step(rk_step)
            |> call_event_fn(options)
            |> interpolate_points(options)
            |> call_output_fn(options.output_fn_adapter)
          else
            step
            |> bump_error_count(options.max_number_of_errors)
          end

        dt_new =
          compute_next_timestep(step.dt_new, rk_step.error_estimate, options.order, step.t_at_start_of_step, t_end, options)

        step = %{step | dt_new: dt_new} |> possibly_delay_playback_speed(options.speed)

        {step, t_end, options}
      end

    updated_step |> record_elapsed_time()
  end

  # Printing example - paste in where necessary:
  # step =
  #   if dt_last == Nx.f64(0.6746869564907434) do
  #     {step, _} =
  #       hook({step, options.output_fn_adapter}, fn {s, adapter} ->
  #         {t, x} = s.output_t_and_x
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
        i_step: step.i_step + 1
    }
  end

  @spec merge_rk_step_into_integration_step(IntegrationStep.t(), RungeKutta.Step.t()) :: IntegrationStep.t()
  defn merge_rk_step_into_integration_step(step, rk_step) do
    %{
      step
      | rk_step: rk_step,
        t_at_start_of_step: rk_step.t_new,
        x_at_start_of_step: rk_step.x_new
    }
  end

  defn interpolate_points(step, options) do
    # This is actually more of an if statement or a case statement
    step
    |> interpolate_intermediate_points(options)
    |> interpolate_fixed_points(options)
  end

  defn interpolate_intermediate_points(step, options) do
    if options.refine == 1 do
      %{step | output_t_and_x: {step.rk_step.t_new, step.rk_step.x_new}}
    else
      {t_add, x_out} = Step.interpolate_multiple_points(step.interpolate_fn, step.rk_step, options)
      %{step | output_t_and_x: {t_add, x_out}}
    end
  end

  defn interpolate_fixed_points(step, _options) do
    step
  end

  @spec call_output_fn(IntegrationStep.t(), ExternalFnAdapter.t()) :: IntegrationStep.t()
  defn call_output_fn(step, output_fn_adapter) do
    {step, _} =
      hook({step, output_fn_adapter}, fn {s, adapter} ->
        # Possibly add a toggle to send the entire step and not just the point?
        {t, x} = s.output_t_and_x
        t_list = Utils.vector_as_list(t)
        x_list = Utils.columns_as_list(x, 0)

        points =
          Enum.zip(t_list, x_list)
          |> Enum.map(fn {t_i, x_i} ->
            %Point{t: t_i, x: x_i}
          end)

        adapter.external_fn.(points)
        {s, adapter}
      end)

    step
  end

  defn possibly_delay_playback_speed(step, _speed) do
    step
  end

  defn call_event_fn(step, _options) do
    # Psuedo-code: first check if an event function is present
    called_fn_output = Nx.u8(1)
    event_happened? = called_fn_output

    if event_happened? do
      step
    else
      step
    end
  end

  # Base zero_tolerance on precision?
  @zero_tolerance 1.0e-07

  @spec continue_stepping?(IntegrationStep.t(), Nx.t()) :: Nx.t()
  defnp continue_stepping?(step, t_end) do
    # Also check the step's status here

    #    if    close to end time                                  or past end time
    if Nx.abs(step.t_at_start_of_step - t_end) <= @zero_tolerance or step.t_at_start_of_step > t_end do
      Nx.u8(0)
    else
      Nx.u8(1)
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

    {step, _options} =
      if step.error_count > max_number_of_errors do
        hook({step, max_number_of_errors}, fn {s, max_errors} ->
          raise MaxErrorsExceededError,
            message: "Too many errors",
            error_count: s.error_count,
            max_number_of_errors: max_errors,
            step: s

          {s, max_errors}
        end)
      else
        {step, max_number_of_errors}
      end

    step
  end

  @spec record_elapsed_time(IntegrationStep.t()) :: IntegrationStep.t()
  defn record_elapsed_time(step) do
    %{step | elapsed_time_μs: timestamp_μs() - step.start_timestamp_μs}
  end
end
