defmodule Integrator.OdeEventHandler do
  @moduledoc """
  Perhaps these functions get moved into AdaptiveStepsize?
  """
  alias Integrator.AdaptiveStepsize.ComputedStep

  # alias Integrator.AdaptiveStepsize
  alias Integrator.Utils
  alias Integrator.NonlinearEqnRoot

  def call_event_fn(event_fn, step, interpolate_fn, opts \\ []) do
    result = event_fn.(step.t_new, step.x_new)

    case result.status do
      :continue -> :continue
      :halt -> {:halt, compute_new_event_fn_step(event_fn, step, interpolate_fn, opts)}
    end
  end

  def compute_new_event_fn_step(event_fn, step, interpolate_fn, opts) do
    zero_fn = fn t ->
      x = interpolate_one_point(t, step, interpolate_fn)
      event_fn.(t, x) |> Map.get(:value) |> Nx.to_number()
    end

    root = NonlinearEqnRoot.find_zero(zero_fn, [Nx.to_number(step.t_old), Nx.to_number(step.t_new)], opts)
    x_new = interpolate_one_point(root.x, step, interpolate_fn)
    %ComputedStep{t_new: root.x, x_new: x_new, k_vals: step.k_vals, options_comp: step.options_comp}
  end

  def interpolate_one_point(t_new, step, interpolate_fn) do
    tadd = Nx.tensor(t_new)

    t = Nx.stack([step.t_old, step.t_new])
    x = Nx.stack([step.x_old, step.x_new]) |> Nx.transpose()

    x_out = interpolate_fn.(t, x, step.k_vals, tadd)
    x_out |> Utils.columns_as_list(0, 0) |> Enum.reverse() |> List.first()
  end
end
