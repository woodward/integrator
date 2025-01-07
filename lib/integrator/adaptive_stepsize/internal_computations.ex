defmodule Integrator.AdaptiveStepsize.InternalComputations do
  @moduledoc """
  Internal computations for `AdaptiveStepsize`
  """

  import Nx.Defn

  alias Integrator.RungeKutta
  alias Integrator.AdaptiveStepsizeRefactor
  alias Integrator.AdaptiveStepsize.MaxErrorsExceededError

  defn integrate_step(step_start, t_end, options) do
    {updated_step, _t_end, _options} =
      while {step = step_start, t_end, options}, finished?(step, t_end) do
        rk_step = RungeKutta.Step.compute_step(step.rk_step, step.dt_new, step.stepper_fn, step.ode_fn, options)
        step = step |> increment_compute_counter()

        step =
          if error_less_than_one?(rk_step) do
            step
            |> reset_error_count_to_zero()
            |> increment_counters()
            |> merge_rk_step_into_integration_step(rk_step)
            |> call_event_fn()
            |> interpolate_intermediate_points()
            |> interpolate_fixed_points()
            |> call_output_fn()
            |> compute_next_success_timestep()
            |> possibly_delay_playback_speed()
          else
            step
            |> bump_error_count(options)
            |> compute_next_error_timestep()
          end

        dt_new = compute_next_timestep(step.dt_new, rk_step.error_estimate, options.order, rk_step.t_new, t_end, options)
        step = %{step | dt_new: dt_new}
        # Needs to be converted to Nx:
        # step = %{step | dt: dt} |> delay_simulation(opts[:speed])

        {step, t_end, options}
      end

    updated_step

    # ------------------------------------
    # Old code:
    # # wrapper around compute_step_nx:
    # DONE new_step = compute_step(step, stepper_fn, ode_fn, opts)

    # DONE step = step |> increment_compute_counter()

    # # could easily be made into Nx:
    # step =
    # DONE  if less_than_one?(new_step.error_estimate) do
    # DONE    step
    # DONE    |> increment_and_reset_counters()
    #     |> merge_new_step(new_step)
    #     |> call_event_fn(opts[:event_fn], opts[:zero_fn], interpolate_fn, opts)
    #     |> interpolate(interpolate_fn, opts[:refine])
    #     |> store_resuts(opts[:store_results?])
    #     |> call_output_fn(opts[:output_fn])
    #   else
    # DONE     bump_error_count(step, opts)
    # DONE   end

    # # This is Nx:
    # dt = compute_next_timestep(step.dt, new_step.error_estimate, order, step.t_new, t_end, opts)

    # # Needs to be converted to Nx:
    # step = %{step | dt: dt} |> delay_simulation(opts[:speed])

    # step
    # # recursive call:
    # |> step(t_next(step, dt), t_end, halt?(step), stepper_fn, interpolate_fn, ode_fn, order, opts)
  end

  defn error_less_than_one?(rk_step) do
    rk_step.error_estimate < 1.0
  end

  defn reset_error_count_to_zero(step) do
    %{step | error_count: 0}
  end

  defn increment_counters(step) do
    %{
      step
      | count_loop__increment_step: step.count_loop__increment_step + 1,
        i_step: step.i_step + 1
    }
  end

  defn merge_rk_step_into_integration_step(step, rk_step) do
    %{
      step
      | rk_step: rk_step,
        t_at_start_of_step: rk_step.t_new,
        x_at_start_of_step: rk_step.x_new
    }
  end

  defn interpolate_intermediate_points(step) do
    step
  end

  defn interpolate_fixed_points(step) do
    step
  end

  defn call_output_fn(step) do
    step
  end

  defn possibly_delay_playback_speed(step) do
    step
  end

  defn call_event_fn(step) do
    step
  end

  defn compute_next_success_timestep(step) do
    step
  end

  defn compute_next_error_timestep(step) do
    step
  end

  defn my_print_value(step, value) do
    step =
      hook(step, fn s ->
        IO.puts("foooo")
        s
      end)

    # {step, _value} =
    #   hook({step, value}, fn {s, v} ->
    #     IO.puts("foooo")
    #     s
    #   end)

    step
  end

  defn print_computing_iteration_type(z) do
    hook(z, fn step ->
      IO.puts("Computing iteration type")
      # IO.puts("Computing iteration type #{inspect(Nx.to_number(step.iteration_type))}")
      step
    end)

    # z
  end

  # Base zero_tolerance on precision?
  @zero_tolerance 1.0e-07

  @spec finished?(AdaptiveStepsizeRefactor.t(), Nx.t()) :: Nx.t()
  defnp finished?(step, t_end) do
    # if Nx.abs(step.t_at_start_of_step - t_end < @zero_tolerance) or step.t_at_start_of_step > t_end do
    #   Nx.u8(1)
    # else
    #   Nx.u8(0)
    # end
    step.count_cycles__compute_step < 78
  end

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
  @spec compute_next_timestep(Nx.t(), Nx.t(), integer(), Nx.t(), Nx.t(), Keyword.t()) :: Nx.t()
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

  @spec bump_error_count(AdaptiveStepsizeRefactor.t(), Keyword.t()) :: AdaptiveStepsizeRefactor.t()
  defnp bump_error_count(step, options) do
    step = %{step | error_count: step.error_count + 1}

    {step, _options} =
      if step.error_count > options.max_number_of_errors do
        hook({step, options}, fn {s, opts} ->
          raise MaxErrorsExceededError,
            message: "Too many errors",
            error_count: s.error_count,
            max_number_of_errors: opts.max_number_of_errors,
            step: s

          {s, opts}
        end)
      else
        {step, options}
      end

    step
  end
end
