defmodule Integrator.Integration do
  @moduledoc """
  A genserver which holds the simulation state for one particular integration.  This GenServer basically
  wraps the pure module functions contained in `Integrator` and `Integrator.AdaptiveStepsize`.
  """

  use GenServer

  alias Integrator.AdaptiveStepsize
  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.InternalComputations
  alias Integrator.AdaptiveStepsize.NxOptions
  alias Integrator.DataCollector
  alias Integrator.Point
  alias Integrator.RungeKutta

  @behaviour DataCollector

  import Integrator.Utils, only: [timestamp_μs: 0]

  @genserver_options [:name, :timeout, :debug, :spawn_opt, :hibernate_after]

  @type integration_status :: :initialized | :running | :paused | :completed

  @type t :: %__MODULE__{
          step: IntegrationStep.t(),
          t_end: Nx.t(),
          options: AdaptiveStepsize.NxOptions.t(),
          caller: GenServer.from() | nil,
          status: integration_status(),
          data: [Point.t()]
        }

  defstruct [
    :step,
    :t_end,
    :options,
    :caller,
    :status,
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
  def run_async(pid), do: GenServer.cast(pid, :run_async)

  @spec pause(GenServer.server()) :: any()
  def pause(pid), do: GenServer.cast(pid, :pause)

  @spec continue(GenServer.server()) :: any()
  def continue(pid), do: GenServer.cast(pid, :continue)

  @spec run(GenServer.server()) :: :ok | {:error, String.t()}
  def run(pid), do: GenServer.call(pid, :run)

  @spec step(GenServer.server()) :: {:ok, IntegrationStep.t()}
  def step(pid), do: GenServer.call(pid, :step)

  @spec can_continue_stepping?(GenServer.server()) :: boolean()
  def can_continue_stepping?(pid), do: GenServer.call(pid, :can_continue_stepping?)

  @spec get_status(GenServer.server()) :: [Point.t()]
  def get_status(pid), do: GenServer.call(pid, :get_status)

  @spec get_step(GenServer.server()) :: IntegrationStep.t()
  def get_step(pid), do: GenServer.call(pid, :get_step)

  @spec get_options(GenServer.server()) :: NxOptions.t()
  def get_options(pid), do: GenServer.call(pid, :get_options)

  @impl DataCollector
  def add_data(pid, point), do: GenServer.cast(pid, {:add_data, point})

  @impl DataCollector
  def get_data(pid), do: GenServer.call(pid, :get_data)

  @impl DataCollector
  def pop_data(pid), do: GenServer.call(pid, :pop_data)

  @impl DataCollector
  def get_last_n_data(pid, number_of_data), do: GenServer.call(pid, {:get_last_n_data, number_of_data})

  # ------------------------------------------------------------------------------------------------
  # Callbacks:

  @impl GenServer
  def init(args) do
    [ode_fn, t_start, t_end, x0, opts] = args
    {integration_opts, integrator_opts} = split_opts(opts)
    integration_opts = integration_opts |> NimbleOptions.validate!(options_schema_integration_only())
    integrator_opts = integrator_opts |> add_data_collector(self(), integration_opts[:store_data_in_genserver?])
    {initial_step, t_end, options} = Integrator.setup_all(ode_fn, t_start, t_end, x0, timestamp_μs(), integrator_opts)
    AdaptiveStepsize.broadcast_initial_point(initial_step, options)

    {:ok, %__MODULE__{step: initial_step, t_end: t_end, options: options, status: :initialized}}
  end

  @impl GenServer
  def handle_cast(:run_async, state) do
    Process.send(self(), :step, [])
    {:noreply, %{state | status: :running}}
  end

  def handle_cast(:pause, state) do
    {:noreply, %{state | status: :paused}}
  end

  def handle_cast(:continue, state) do
    Process.send(self(), :step, [])
    {:noreply, %{state | status: :running}}
  end

  def handle_cast({:add_data, point}, state) do
    state = %{state | data: List.wrap(point) ++ state.data}
    {:noreply, state}
  end

  @impl GenServer
  def handle_call(:run, from, state) do
    Process.send(self(), :step, [])
    {:noreply, %{state | caller: from, status: :running}}
  end

  def handle_call(:step, _from, %{step: step, t_end: t_end, options: options} = state) do
    step = %{step | step_timestamp_μs: timestamp_μs()}
    step = InternalComputations.compute_integration_step(step, t_end, options)
    {:reply, {:ok, step}, %{state | step: step, status: :paused}}
  end

  def handle_call(:can_continue_stepping?, _from, %{step: step, t_end: t_end} = state) do
    {:reply, can_continue_stepping?(step, t_end), state}
  end

  def handle_call(:get_data, _from, state) do
    {:reply, state.data |> Enum.reverse(), state}
  end

  def handle_call(:pop_data, _from, state) do
    {:reply, state.data |> Enum.reverse(), %{state | data: []}}
  end

  def handle_call({:get_last_n_data, number_of_data}, _from, state) do
    {:reply, state.data |> Enum.take(number_of_data) |> Enum.reverse(), state}
  end

  def handle_call(:get_status, _from, state) do
    {:reply, state.status, state}
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
    {:noreply, %{state | status: :completed}}
  end

  def handle_info(:step, %{step: step, t_end: t_end, options: options, status: status} = state) do
    step =
      if can_continue_running?(status, step, t_end) do
        step = %{step | step_timestamp_μs: timestamp_μs()}
        step = InternalComputations.compute_integration_step(step, t_end, options)
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

  # ------------------------------------------------------------------------------------------------

  @spec split_opts(Keyword.t()) :: {Keyword.t(), Keyword.t()}
  defp split_opts(opts) do
    integration_opt_keys = options_schema_integration_only() |> Map.get(:schema) |> Keyword.keys()
    opts |> Keyword.split(integration_opt_keys)
  end

  @spec add_data_collector(Keyword.t(), GenServer.server(), boolean()) :: Keyword.t()
  defp add_data_collector(integrator_opts, _pid, false = _store_data_in_genserver?), do: integrator_opts

  defp add_data_collector(integrator_opts, pid, true = _store_data_in_genserver?) do
    existing_output_fn = Keyword.get(integrator_opts, :output_fn)

    new_output_fn = fn point ->
      existing_output_fn.(point)
      add_data(pid, point)
    end

    integrator_opts |> Keyword.merge(output_fn: new_output_fn)
  end

  @spec can_continue_stepping?(IntegrationStep.t(), Nx.t()) :: boolean()
  defp can_continue_stepping?(step, t_end) do
    Nx.to_number(InternalComputations.continue_stepping?(step, t_end)) == 1
  end

  @spec can_continue_running?(integration_status(), IntegrationStep.t(), Nx.t()) :: boolean()
  defp can_continue_running?(status, step, t_end) do
    status == :running && can_continue_stepping?(step, t_end)
  end
end
