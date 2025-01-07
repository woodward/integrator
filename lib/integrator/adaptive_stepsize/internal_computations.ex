defmodule Integrator.AdaptiveStepsize.InternalComputations do
  @moduledoc """
  Internal computations for `AdaptiveStepsize`
  """

  import Nx.Defn

  alias Integrator.RungeKutta

  defn integrate_step(step_start, t_end, options) do
    {updated_step, _t_end, _options} =
      while {step = step_start, t_end, options}, finished?(step) do
        rk_step = RungeKutta.Step.compute_step(step.rk_step, step.dt_new, step.stepper_fn, step.ode_fn, options)
        step = step |> increment_counters()
        step = %{step | rk_step: rk_step}

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
    #   if less_than_one?(new_step.error_estimate) do
    #     step
    #     |> increment_and_reset_counters()
    #     |> merge_new_step(new_step)
    #     |> call_event_fn(opts[:event_fn], opts[:zero_fn], interpolate_fn, opts)
    #     |> interpolate(interpolate_fn, opts[:refine])
    #     |> store_resuts(opts[:store_results?])
    #     |> call_output_fn(opts[:output_fn])
    #   else
    #     bump_error_count(step, opts)
    #   end

    # # This is Nx:
    # dt = compute_next_timestep(step.dt, new_step.error_estimate, order, step.t_new, t_end, opts)

    # # Needs to be converted to Nx:
    # step = %{step | dt: dt} |> delay_simulation(opts[:speed])

    # step
    # # recursive call:
    # |> step(t_next(step, dt), t_end, halt?(step), stepper_fn, interpolate_fn, ode_fn, order, opts)
  end

  defnp finished?(step) do
    # when abs(t_old - t_end) < @zero_tolerance or t_old > t_end   ->  halt
    # when status == :halt       -> halt

    step.count_cycles__compute_step < 78
  end

  defnp increment_counters(step) do
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
end
