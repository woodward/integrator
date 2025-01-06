defmodule Integrator.AdaptiveStepsize.InternalComputations do
  @moduledoc """
  Internal computations for `AdaptiveStepsize`
  """

  import Nx.Defn

  defn integrate_step(step_start, _stepper_fn, _interpolate_fn, _ode_fn, _t_start, _t_end, _initial_tstep, _x0, _options) do
    {updated_step} =
      while {step = step_start}, finished?(step) do
        step = step |> increment_counters()
        {step}
      end

    updated_step
  end

  defnp finished?(step) do
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
  defn compute_next_timestep(dt, error, order, t_old, t_end, opts) do
    nx_type = opts[:type]

    # Avoid divisions by zero:
    # error = error + Nx.Constants.epsilon(nx_type)
    error = error + Nx.Constants.epsilon(nx_type)

    # Octave:
    #   dt *= min (facmax, max (facmin, fac * (1 / err)^(1 / (order + 1))));

    one = Nx.tensor(1.0, type: nx_type)
    foo = factor(order, nx_type) * (one / error) ** exponent(order, nx_type)
    dt = dt * min(Nx.tensor(@stepsize_factor_max, type: nx_type), max(Nx.tensor(@stepsize_factor_min, type: nx_type), foo))
    dt = min(Nx.abs(dt), opts[:max_step])

    # Make sure we don't go past t_end:
    min(Nx.abs(dt), Nx.abs(t_end - t_old))
  end
end
