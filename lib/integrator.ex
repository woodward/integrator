defmodule Integrator do
  @moduledoc """
  A library for solving non-stiff ordinary differential equations (ODEs).

  `Integrator` uses either the Dormand-Prince 4/5 Runge Kutta algorithm, or the Bogacki-Shampine 2/3
  Runge Kutta algorithm.  It is intended that the user only needs to call `Integrator.integrate/4`, and
  the `Integrator.AdaptiveStepsize` and `Integrator.RungeKutta` modules are only exposed for advanced
  users who want to use the underlying algorithms directly.
  """

  alias Integrator.AdaptiveStepsize
  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.NxOptions
  alias Integrator.RungeKutta
  alias Integrator.RungeKutta.BogackiShampine23
  alias Integrator.RungeKutta.DormandPrince45

  options = [
    initial_step: [
      doc: """
      The initial stepsize. If not provided, a stepsize will be chosen automatically. Can be a float
      or an Nx tensor.
      """,
      type: {:or, [:float, :any]}
    ],
    integrator: [
      doc: """
      The integrator to use. Two different Runge-Kutta implentations are supported:
      `Integrator.RungeKutta.DormandPrince45` and `Integrator.RungeKutta.BogackiShampine23`.
      """,
      type: {:in, [DormandPrince45, BogackiShampine23]},
      default: DormandPrince45
    ]
  ]

  @options_schema_integrator_only NimbleOptions.new!(options)
  def options_schema_integrator_only, do: @options_schema_integrator_only

  # @options_schema NimbleOptions.new!(AdaptiveStepsize.options_schema().schema |> Keyword.merge(options))

  @doc """
  Integrates an ODE function using either the Dormand-Prince45 method or the Bogacki-Shampine23 method.

  ## Options

  #{NimbleOptions.docs(@options_schema_integrator_only)}

  ### Additional Options

  Also see the options for these functions which are passed through:

  * `Integrator.NonLinearEqnRoot.find_zero/5`
  * `Integrator.AdaptiveStepsize.integrate/8`

  """
  @spec integrate(
          ode_fn :: RungeKutta.ode_fn_t(),
          t_start :: Nx.t() | float(),
          t_end :: Nx.t() | float(),
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: IntegrationStep.t()
  def integrate(ode_fn, t_start, t_end, x0, opts \\ []) do
    {integrator_mod, initial_step, order, remaining_opts} = setup(opts)

    AdaptiveStepsize.integrate(
      &integrator_mod.integrate/6,
      &integrator_mod.interpolate/4,
      ode_fn,
      t_start,
      t_end,
      initial_step,
      x0,
      order,
      remaining_opts
    )
  end

  @spec setup(Keyword.t()) :: {module(), Nx.t() | float(), integer(), Keyword.t()}
  def setup(opts) do
    local_opt_keys = options_schema_integrator_only() |> Map.get(:schema) |> Keyword.keys()
    {local_opts, remaining_opts} = Keyword.split(opts, local_opt_keys)
    local_opts = local_opts |> NimbleOptions.validate!(@options_schema_integrator_only)

    integrator_mod = Keyword.get(local_opts, :integrator)
    initial_step = Keyword.get(local_opts, :initial_step)
    order = integrator_mod.order()

    {integrator_mod, initial_step, order, remaining_opts}
  end

  @spec setup_all(
          ode_fn :: RungeKutta.ode_fn_t(),
          t_start :: Nx.t(),
          t_end :: Nx.t(),
          x0 :: Nx.t(),
          start_timestamp_ms :: Nx.t(),
          opts :: Keyword.t()
        ) :: {IntegrationStep.t(), Nx.t(), NxOptions.t()}
  def setup_all(ode_fn, t_start, t_end, x0, timestamp_μs, opts) do
    {integrator_mod, initial_tstep, order, remaining_opts} = setup(opts)

    stepper_fn = &integrator_mod.integrate/6
    interpolate_fn = &integrator_mod.interpolate/4

    AdaptiveStepsize.setup(
      stepper_fn,
      interpolate_fn,
      ode_fn,
      t_start,
      t_end,
      initial_tstep,
      x0,
      order,
      timestamp_μs,
      remaining_opts
    )
  end
end
