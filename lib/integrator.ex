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
  alias Integrator.RungeKutta
  alias Integrator.RungeKutta.{BogackiShampine23, DormandPrince45}

  @integrator_options %{
    ode45: DormandPrince45,
    ode23: BogackiShampine23
  }

  @default_refine_opts %{
    ode45: 4,
    ode23: 1
  }

  options = [
    initial_step: [],
    integrator: [
      type: {:in, [:ode45, :ode23]},
      default: :ode45
    ]
  ]

  @options_schema NimbleOptions.new!(AdaptiveStepsize.options_schema().schema |> Keyword.merge(options))

  @doc """
  Integrates an ODE function using either the Dormand-Prince45 method or the Bogacki-Shampine23 method
  """
  @spec integrate(
          ode_fn :: RungeKutta.ode_fn_t(),
          t_start_t_end :: Nx.t() | [float() | Nx.t()],
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: AdaptiveStepsize.t()
  def integrate(ode_fn, t_start_t_end, x0, opts \\ []) do
    opts =
      opts
      |> NimbleOptions.validate!(@options_schema)
      |> Keyword.put_new_lazy(:type, fn -> Nx.type(x0) |> Nx.Type.to_string() |> String.to_atom() end)
      |> AdaptiveStepsize.abs_rel_norm_opts()

    opts = opts |> Keyword.put_new_lazy(:refine, fn -> Map.get(@default_refine_opts, opts[:integrator]) end)

    integrator_mod =
      Map.get_lazy(@integrator_options, opts[:integrator], fn ->
        raise "Currently only DormandPrince45 (ode45) and BogackiShampine23 (ode23) are supported"
      end)

    order = integrator_mod.order()
    {t_start, t_end, fixed_times} = parse_start_end(t_start_t_end)

    initial_step =
      Keyword.get_lazy(opts, :initial_step, fn ->
        AdaptiveStepsize.starting_stepsize(order, ode_fn, t_start, x0, opts[:abs_tol], opts[:rel_tol], opts)
      end)

    AdaptiveStepsize.integrate(
      &integrator_mod.integrate/6,
      &integrator_mod.interpolate/4,
      ode_fn,
      t_start,
      t_end,
      fixed_times,
      initial_step,
      x0,
      order,
      only_adaptive_stepsize_opts(opts)
    )
  end

  @spec only_adaptive_stepsize_opts(Keyword.t()) :: Keyword.t()
  defp only_adaptive_stepsize_opts(opts) do
    adaptive_stepsize_opts_keys = AdaptiveStepsize.option_keys()
    opts |> Keyword.filter(fn {key, _value} -> key in adaptive_stepsize_opts_keys end)
  end

  @spec parse_start_end([float() | Nx.t()] | Nx.t()) :: {Nx.t(), Nx.t(), [Nx.t()] | nil}
  defp parse_start_end([t_start, t_end]), do: {t_start, t_end, nil}

  defp parse_start_end(t_range) do
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
end
