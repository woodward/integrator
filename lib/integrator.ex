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
  alias Integrator.RungeKutta.{BogackiShampine23, DormandPrince45}

  @integrator_options %{
    ode45: DormandPrince45,
    ode23: BogackiShampine23
  }

  @default_refine_opts %{
    ode45: 4,
    ode23: 1
  }

  @default_opts [integrator: :ode45]

  @spec integrate(fun(), Nx.t() | [float], Nx.t(), Keyword.t()) :: AdaptiveStepsize.t()
  def integrate(ode_fn, t_start_t_end, x0, opts \\ []) do
    opts = opts |> merge_default_opts()

    integrator_mod =
      Map.get_lazy(@integrator_options, opts[:integrator], fn ->
        raise "Currently only DormandPrince45 (ode45) and BogackiShampine23 (ode23) are supported"
      end)

    order = integrator_mod.order()
    {t_start, t_end, fixed_times} = parse_start_end(t_start_t_end)
    initial_tstep = Utils.starting_stepsize(order, ode_fn, t_start, x0, opts[:abs_tol], opts[:rel_tol], norm_control: false)

    AdaptiveStepsize.integrate(
      &integrator_mod.integrate/5,
      &integrator_mod.interpolate/4,
      ode_fn,
      t_start,
      t_end,
      fixed_times,
      Nx.to_number(initial_tstep),
      x0,
      order,
      opts
    )
  end

  @spec parse_start_end([float() | Nx.t()] | Nx.t()) :: {float(), float(), [float()] | nil}
  defp parse_start_end([t_start, t_end]), do: {t_start, t_end, _fixed_times = nil}

  defp parse_start_end(t_start_and_t_end) do
    t_start = t_start_and_t_end[0] |> Nx.to_number()
    {length} = Nx.shape(t_start_and_t_end)

    # The following Nx.as_type(:f32) is a HACK as I think there's a bug in Nx.linspace():
    t_end = t_start_and_t_end[length - 1] |> Nx.as_type(:f32) |> Nx.to_number()

    fixed_times = t_start_and_t_end |> Nx.to_list() |> Enum.map(&Nx.to_number(&1))
    {t_start, t_end, fixed_times}
  end

  @spec set_default_refine_opt(Keyword.t()) :: Keyword.t()
  defp set_default_refine_opt(opts) do
    if opts[:refine] do
      opts
    else
      default_refine_for_integrator = Map.get(@default_refine_opts, opts[:integrator])
      opts |> Keyword.merge(refine: default_refine_for_integrator)
    end
  end

  @spec merge_default_opts(Keyword.t()) :: Keyword.t()
  defp merge_default_opts(user_specified_opts) do
    @default_opts |> Keyword.merge(Utils.default_opts()) |> Keyword.merge(user_specified_opts) |> set_default_refine_opt()
  end
end
