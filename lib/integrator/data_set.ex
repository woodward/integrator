defmodule Integrator.DataSet do
  @moduledoc """
  Collects data from an integration (or when finding a root).
  Data is stored in reverse order in the genserver, and then its order is reversed when returned
  to the caller.
  """

  use GenServer

  alias Integrator.DataCollector

  @behaviour DataCollector

  @spec start_link(Keyword.t()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, :ok, opts)
  end

  @impl DataCollector
  def add_data(pid, data_point) do
    GenServer.cast(pid, {:add_data, data_point})
  end

  @impl DataCollector
  def get_data(pid) do
    GenServer.call(pid, :get_data)
  end

  @impl DataCollector
  def pop_data(pid) do
    GenServer.call(pid, :pop_data)
  end

  @impl DataCollector
  def get_last_n_data(pid, number_of_data) do
    GenServer.call(pid, {:get_last_n_data, number_of_data})
  end

  # ------------------------------------------------------------------------------------------------

  @impl true
  def init(_args) do
    {:ok, %{data: []}}
  end

  @impl true
  def handle_cast({:add_data, data_points}, state) when is_list(data_points) do
    new_data = Enum.reverse(data_points) ++ state.data
    {:noreply, %{state | data: new_data}}
  end

  def handle_cast({:add_data, data_point}, state) do
    new_data = [data_point | state.data]
    {:noreply, %{state | data: new_data}}
  end

  @impl true
  def handle_call(:get_data, _from, state) do
    {:reply, state.data |> Enum.reverse(), state}
  end

  def handle_call(:pop_data, _from, state) do
    {:reply, state.data |> Enum.reverse(), %{state | data: []}}
  end

  def handle_call({:get_last_n_data, number_of_data}, _from, state) do
    {:reply, state.data |> Enum.take(number_of_data) |> Enum.reverse(), state}
  end
end
