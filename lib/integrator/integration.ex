defmodule Integrator.Integration do
  @moduledoc """
  A genserver which holds the simulation state for one particular integration
  """

  use GenServer

  alias Integrator.AdaptiveStepsize
  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.InternalComputations
  alias Integrator.AdaptiveStepsize.NxOptions
  alias Integrator.Point
  alias Integrator.RungeKutta

  import Integrator.Utils, only: [timestamp_μs: 0]

  @genserver_options [:name, :timeout, :debug, :spawn_opt, :hibernate_after]

  @type t :: %__MODULE__{
          step: IntegrationStep.t(),
          t_end: Nx.t(),
          options: AdaptiveStepsize.NxOptions.t(),
          caller: GenServer.from() | nil,
          data: [Point.t()]
        }

  defstruct [
    :step,
    :t_end,
    :options,
    :caller,
    data: []
  ]

  options = [
    store_data_in_genserver?: [
      doc: """
      Store the output data in the genserver.
      """,
      type: :boolean,
      default: false
    ]
  ]

  @options_schema_integration_only NimbleOptions.new!(options)
  def options_schema_integration_only, do: @options_schema_integration_only

  @spec start_link(
          ode_fn :: RungeKutta.ode_fn_t(),
          t_start :: Nx.t() | float(),
          t_end :: Nx.t() | float(),
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: GenServer.on_start()
  def start_link(ode_fn, t_start, t_end, x0, opts \\ []) do
    {genserver_opts, integrator_opts} = opts |> Keyword.split(@genserver_options)
    GenServer.start_link(__MODULE__, [ode_fn, t_start, t_end, x0, integrator_opts], genserver_opts)
  end

  @spec run_async(GenServer.server()) :: any()
  def run_async(pid) do
    GenServer.cast(pid, :run_async)
  end

  @spec add_data_point(GenServer.server(), Point.t()) :: any()
  def add_data_point(pid, point) do
    GenServer.cast(pid, {:add_data_point, point})
  end

  @spec run(GenServer.server()) :: :ok | {:error, String.t()}
  def run(pid) do
    GenServer.call(pid, :run)
  end

  @spec step(GenServer.server()) :: {:ok, IntegrationStep.t()}
  def step(pid) do
    GenServer.call(pid, :step)
  end

  @spec can_continue_stepping?(GenServer.server()) :: boolean()
  def can_continue_stepping?(pid) do
    GenServer.call(pid, :can_continue_stepping?)
  end

  @spec get_data(GenServer.server()) :: [Point.t()]
  def get_data(pid) do
    GenServer.call(pid, :get_data)
  end

  @spec get_step(GenServer.server()) :: IntegrationStep.t()
  def get_step(pid) do
    GenServer.call(pid, :get_step)
  end

  @spec get_options(GenServer.server()) :: NxOptions.t()
  def get_options(pid) do
    GenServer.call(pid, :get_options)
  end

  # ------------------------------------------------------------------------------------------------

  @impl GenServer
  def init(args) do
    [ode_fn, t_start, t_end, x0, opts] = args
    {integration_opts, integrator_opts} = split_opts(opts)
    integration_opts = integration_opts |> NimbleOptions.validate!(options_schema_integration_only())
    integrator_opts = integrator_opts |> add_data_collector(self(), integration_opts[:store_data_in_genserver?])
    {initial_step, t_end, options} = Integrator.setup_all(ode_fn, t_start, t_end, x0, timestamp_μs(), integrator_opts)
    AdaptiveStepsize.broadcast_initial_point(initial_step, options)

    {:ok, %__MODULE__{step: initial_step, t_end: t_end, options: options}}
  end

  @impl GenServer
  def handle_cast(:run_async, state) do
    Process.send(self(), :step, [])
    {:noreply, state}
  end

  def handle_cast({:add_data_point, point}, state) do
    state = %{state | data: List.wrap(point) ++ state.data}
    {:noreply, state}
  end

  @impl GenServer
  def handle_call(:run, from, state) do
    Process.send(self(), :step, [])
    {:noreply, Map.put(state, :caller, from)}
  end

  def handle_call(:step, _from, %{step: step, t_end: t_end, options: options} = state) do
    step = %{step | step_timestamp_μs: timestamp_μs()}
    {step, _t_end, options} = InternalComputations.compute_integration_step(step, t_end, options)
    {:reply, {:ok, step}, %{state | step: step, options: options}}
  end

  def handle_call(:can_continue_stepping?, _from, %{step: step, t_end: t_end} = state) do
    {:reply, can_continue_stepping?(step, t_end), state}
  end

  def handle_call(:get_data, _from, state) do
    {:reply, state.data |> Enum.reverse(), state}
  end

  def handle_call(:get_step, _from, state) do
    {:reply, state.step, state}
  end

  def handle_call(:get_options, _from, state) do
    {:reply, state.options, state}
  end

  @impl GenServer
  def handle_info(:run_completed, state) do
    result = state.step.status_integration |> IntegrationStep.status_integration()
    if state.caller, do: GenServer.reply(state.caller, result)
    {:noreply, state}
  end

  def handle_info(:step, %{step: step, t_end: t_end, options: options} = state) do
    step =
      if can_continue_stepping?(step, t_end) do
        step = %{step | step_timestamp_μs: timestamp_μs()}
        {step, _t_end, options} = InternalComputations.compute_integration_step(step, t_end, options)
        sleep_time_ms = InternalComputations.compute_sleep_time(step, options)

        if sleep_time_ms > 0 do
          Process.send_after(self(), :step, sleep_time_ms)
          step
        else
          Process.send(self(), :step, [])

          step
        end
      else
        Process.send(self(), :run_completed, [])
        step
      end

    {:noreply, %{state | step: step}}
  end

  defp split_opts(opts) do
    integration_opt_keys = options_schema_integration_only() |> Map.get(:schema) |> Keyword.keys()
    opts |> Keyword.split(integration_opt_keys)
  end

  defp add_data_collector(integrator_opts, _pid, false = _store_data_in_genserver?), do: integrator_opts

  defp add_data_collector(integrator_opts, pid, true = _store_data_in_genserver?) do
    output_fn = Keyword.get(integrator_opts, :output_fn)

    new_output_fn = fn point ->
      output_fn.(point)
      add_data_point(pid, point)
    end

    integrator_opts |> Keyword.merge(output_fn: new_output_fn)
  end

  defp can_continue_stepping?(step, t_end) do
    Nx.to_number(InternalComputations.continue_stepping?(step, t_end)) == 1
  end
end
