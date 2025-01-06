defmodule Integrator.AdaptiveStepsize.InternalComputations do
  @moduledoc """
  Internal computations for `AdaptiveStepsize`
  """

  import Nx.Defn

  defn integrate_step(integration, stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, options) do
    integration =
      while {integration = integration}, finished?(integration) do
        integration = integration |> increment_counters()
        {integration}
      end

    integration
  end

  defnp finished?(integration) do
    integration.count_loop__increment_step > 5
  end

  defnp increment_counters(integration) do
    %{integration | count_loop__increment_step: integration.count_loop__increment_step + 1}
  end
end
