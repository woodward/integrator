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

  @default_opts [
    integrator: :ode45
  ]

  @doc """
  Integrates an ODE function using either the Dormand-Prince45 method or the Bogacki-Shampine23 method
  """
  @spec integrate(
          ode_fn :: fun(),
          t_start_t_end :: Nx.t() | [float() | Nx.t()],
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: AdaptiveStepsize.t()
  def integrate(ode_fn, t_start_t_end, x0, opts \\ []) do
    opts = opts |> merge_default_opts()
    {opts, x0} = determine_nx_type(opts, x0)

    integrator_mod =
      Map.get_lazy(@integrator_options, opts[:integrator], fn ->
        raise "Currently only DormandPrince45 (ode45) and BogackiShampine23 (ode23) are supported"
      end)

    order = integrator_mod.order()
    {t_start, t_end, fixed_times} = parse_start_end(t_start_t_end, opts)
    initial_tstep = AdaptiveStepsize.starting_stepsize(order, ode_fn, t_start, x0, opts[:abs_tol], opts[:rel_tol], opts)

    AdaptiveStepsize.integrate(
      &integrator_mod.integrate/5,
      &integrator_mod.interpolate/4,
      ode_fn,
      t_start,
      t_end,
      fixed_times,
      initial_tstep,
      x0,
      order,
      opts
    )
  end

  @spec parse_start_end([float() | Nx.t()] | Nx.t(), Keyword.t()) :: {Nx.t(), Nx.t(), [Nx.t()] | nil}
  defp parse_start_end([t_start, t_end], opts) do
    nx_type = opts[:type]
    {Nx.tensor(t_start, type: nx_type), Nx.tensor(t_end, type: nx_type), _fixed_times = nil}
  end

  defp parse_start_end(t_range, opts) do
    if Utils.type_atom(t_range) != opts[:type] do
      raise ArgumentError, "The Nx type for the time range and the option[:type] do not match"
    end

    t_start = t_range[0]
    {length} = Nx.shape(t_range)

    t_end = t_range[length - 1]

    # Figure out the correct way to do this; there's got to be a better way!
    fixed_times =
      0..(length - 1)
      |> Enum.reduce([], fn i, acc ->
        [t_range[i] | acc]
      end)
      |> Enum.reverse()

    {t_start, t_end, fixed_times}
  end

  @spec merge_default_opts(Keyword.t()) :: Keyword.t()
  defp merge_default_opts(user_specified_opts) do
    @default_opts
    |> Keyword.merge(AdaptiveStepsize.default_opts())
    |> Keyword.merge(user_specified_opts)
    |> set_default_refine_opt(user_specified_opts)
  end

  @spec set_default_refine_opt(Keyword.t(), Keyword.t()) :: Keyword.t()
  defp set_default_refine_opt(opts, user_specified_opts) do
    if user_specified_opts[:refine] do
      opts
    else
      default_refine_for_integrator = Map.get(@default_refine_opts, opts[:integrator])
      opts |> Keyword.merge(refine: default_refine_for_integrator)
    end
  end

  @spec determine_nx_type(Keyword.t(), Nx.t()) :: {Keyword.t(), Nx.t()}
  defp determine_nx_type(opts, x0) do
    case opts[:type] do
      nil -> {Keyword.merge(opts, type: Utils.type_atom(x0)), x0}
      nx_type -> {opts, Nx.as_type(x0, nx_type)}
    end
  end
end
