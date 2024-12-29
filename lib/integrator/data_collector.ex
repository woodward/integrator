defmodule Integrator.DataCollector do
  @moduledoc """
  Collects data from an integration (or when finding a root)
  """

  use GenServer

  @spec start_link(Keyword.t()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, :ok, opts)
  end

  @spec add_data(pid(), Nx.t()) :: :ok
  def add_data(pid, data_point) do
    GenServer.cast(pid, {:add_data, data_point})
  end

  @spec get_data(pid()) :: [Nx.t()]
  def get_data(pid) do
    GenServer.call(pid, :get_data)
  end

  @impl true
  def init(_args) do
    {:ok, %{data: []}}
  end

  @impl true
  def handle_cast({:add_data, data_point}, state) do
    new_data = [data_point | state.data]
    {:noreply, %{state | data: new_data}}
  end

  @impl true
  def handle_call(:get_data, _from, state) do
    {:reply, state.data |> Enum.reverse(), state}
  end
end
