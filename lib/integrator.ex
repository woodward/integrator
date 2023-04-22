defmodule Integrator do
  @moduledoc """
  $$  F = M A $$

  or an inline equation via a single dollar sign, such as $ F = M A $

  Some more equations:

  $$ f(x) =  \\int_{-\\infty}^\\infty
    f\\hat(\\xi)\\,e^{2 \\pi i \\xi x}
    \\,d\\xi $$

  """

  alias Integrator.{AdaptiveStepsize, Utils}
  alias Integrator.RungeKutta.DormandPrince45

  def integrate(ode_fn, t_start, t_end, x0, opts \\ []) do
    merged_opts = default_opts() |> Keyword.merge(Utils.default_opts())
    integrator_mod = integrator_mod(merged_opts)

    stepper_fn = &integrator_mod.integrate/5
    interpolate_fn = &integrator_mod.interpolate/4
    order = integrator_mod.order()

    opts = merged_opts |> Keyword.merge(integrator_mod.default_opts()) |> Keyword.merge(opts)

    initial_tstep = Utils.starting_stepsize(order, ode_fn, t_start, x0, opts[:abs_tol], opts[:rel_tol], norm_control: false)
    AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, [t_start, t_end], Nx.to_number(initial_tstep), x0, order, opts)
  end

  def default_opts() do
    [integrator: :ode45]
  end

  def integrator_mod(opts) do
    case opts[:integrator] do
      :ode45 -> DormandPrince45
      _ -> raise "Currently only DormandPrince45 (ode45) is supported"
    end
  end
end
