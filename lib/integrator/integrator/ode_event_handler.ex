defmodule Integrator.OdeEventHandler do
  @moduledoc false

  defstruct status: :continue

  def call_event_fn(event_fn, t, x, _k_vals, _order, _opts \\ []) do
    result = event_fn.(t, x)
    result.status

    %__MODULE__{status: result.status}
  end
end
