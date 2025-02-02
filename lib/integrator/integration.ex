defmodule Integrator.Integration do
  @moduledoc """
  A genserver which holds the simulation state for one particular integration
  """

  use GenServer

  alias Integrator.AdaptiveStepsize
  alias Integrator.RungeKutta

  import Integrator.Utils, only: [timestamp_μs: 0]

  @genserver_options [:name, :timeout, :debug, :spawn_opt, :hibernate_after]

  defstruct [:step, :t_end, :options]

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

  @impl GenServer
  def init(args) do
    [ode_fn, t_start, t_end, x0, opts] = args
    {initial_step, t_end, options} = Integrator.setup_all(ode_fn, t_start, t_end, x0, timestamp_μs(), opts)
    AdaptiveStepsize.broadcast_initial_point(initial_step, options)

    {:ok, %__MODULE__{step: initial_step, t_end: t_end, options: options}}
  end
end
