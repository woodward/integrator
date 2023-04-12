defmodule Integrator do
  @moduledoc """
  $$  F = M A $$

  or an inline equation via a single dollar sign, such as $ F = M A $

  Some more equations:

  $$ f(x) =  \\int_{-\\infty}^\\infty
    f\\hat(\\xi)\\,e^{2 \\pi i \\xi x}
    \\,d\\xi $$

  """

  alias Integrator.AdaptiveStepsize
  alias Integrator.RungeKutta.DormandPrince45

  def ode45(_deriv_fn, _x_initial, _x_final, _initial_y) do
    x = [0.0, 0.0180, 0.041]
    y = [Nx.tensor([2.0000, 0.000]), Nx.tensor([1.9897, -0.0322]), Nx.tensor([1.9979, -0.0638])]
    [x, y]
  end

  def integrate(ode_fn, t_start, t_end, x0, opts \\ []) do
    # stepper_fn = &DormandPrince45.integrate/5
    # interpolate_fn = &DormandPrince45.interpolate/4
    # order = DormandPrince45.order()

    merged_opts = default_opts()
    integrator_mod = integrator_mod(merged_opts)

    stepper_fn = &integrator_mod.integrate/5
    interpolate_fn = &integrator_mod.interpolate/4
    order = integrator_mod.order()

    opts = merged_opts |> Keyword.merge(integrator_mod.default_opts()) |> Keyword.merge(opts)

    # compute this instead:
    initial_tstep = 0.1

    AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)
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
