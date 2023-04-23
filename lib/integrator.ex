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

  @spec integrate(fun(), float(), float(), Nx.t(), Keyword.t()) :: AdaptiveStepsize.t()
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

  @spec default_opts() :: Keyword.t()
  defp default_opts() do
    [integrator: :ode45]
  end

  @spec integrator_mod(Keyword.t()) :: module()
  defp integrator_mod(opts) do
    case opts[:integrator] do
      :ode45 -> DormandPrince45
      _ -> raise "Currently only DormandPrince45 (ode45) is supported"
    end
  end
end
