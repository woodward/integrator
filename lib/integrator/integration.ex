defmodule Integrator.Integration do
  @moduledoc """
  A genserver which holds the simulation state for one particular integration
  """

  use GenServer

  alias Integrator.AdaptiveStepsize
  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.InternalComputations
  alias Integrator.RungeKutta

  import Integrator.Utils, only: [timestamp_μs: 0]

  @genserver_options [:name, :timeout, :debug, :spawn_opt, :hibernate_after]

  @type t :: %__MODULE__{
          step: IntegrationStep.t(),
          t_end: Nx.t(),
          options: AdaptiveStepsize.NxOptions.t(),
          caller: GenServer.from() | nil
        }

  defstruct [:step, :t_end, :options, :caller]

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

  @spec run(GenServer.server()) :: :ok | {:error, String.t()}
  def run(pid) do
    GenServer.call(pid, :run)
  end

  @impl GenServer
  def init(args) do
    [ode_fn, t_start, t_end, x0, opts] = args
    {initial_step, t_end, options} = Integrator.setup_all(ode_fn, t_start, t_end, x0, timestamp_μs(), opts)
    AdaptiveStepsize.broadcast_initial_point(initial_step, options)

    {:ok, %__MODULE__{step: initial_step, t_end: t_end, options: options}}
  end

  @impl GenServer
  def handle_cast(:run_async, state) do
    Process.send(self(), :step, [])
    {:noreply, state}
  end

  @impl GenServer
  def handle_call(:run, from, state) do
    Process.send(self(), :step, [])
    {:noreply, Map.put(state, :caller, from)}
  end

  @impl GenServer
  def handle_info(:run_completed, state) do
    result = state.step.status_integration |> IntegrationStep.status_integration()
    if state.caller, do: GenServer.reply(state.caller, result)
    {:noreply, state}
  end

  def handle_info(:step, %{step: step, t_end: t_end, options: options} = state) do
    step =
      if Nx.to_number(InternalComputations.continue_stepping?(step, t_end)) == 1 do
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
end
