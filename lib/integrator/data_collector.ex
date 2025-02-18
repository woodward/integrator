defmodule Integrator.DataCollector do
  @moduledoc """
  A behaviour which defines a process which collects data from an integration
  """

  alias Integrator.Point

  defmacro __using__(_opts) do
    quote do
      # on_mount({Integrator.DataCollector, :init_data})
      alias Integrator.DataCollector
      @behaviour DataCollector

      @impl DataCollector
      def add_data(pid, data_set_id, data_point) do
        GenServer.cast(pid, {:add_data, data_point, data_set_id})
      end

      @impl DataCollector
      def get_data(pid, data_set_id) do
        GenServer.call(pid, {:get_data, data_set_id})
      end

      @impl DataCollector
      def pop_data(pid, data_set_id) do
        GenServer.call(pid, {:pop_data, data_set_id})
      end

      @impl DataCollector
      def get_last_n_data(pid, data_set_id, number_of_data) do
        GenServer.call(pid, {:get_last_n_data, number_of_data, data_set_id})
      end

      @impl true
      def handle_cast({:add_data, data_points, caller_pid}, state) do
        existing_data = Map.get(state.data, caller_pid, [])
        new_data = (List.wrap(data_points) |> Enum.reverse()) ++ existing_data
        {:noreply, %{state | data: Map.put(state.data, caller_pid, new_data)}}
      end

      @impl true
      def handle_call({:get_data, caller_pid}, _from, state) do
        {:reply, state.data |> Map.get(caller_pid, []) |> Enum.reverse(), state}
      end

      def handle_call({:pop_data, caller_pid}, _from, state) do
        points = state.data |> Map.get(caller_pid, []) |> Enum.reverse()
        {:reply, points, %{state | data: Map.put(state.data, caller_pid, [])}}
      end

      def handle_call({:get_last_n_data, number_of_data, caller_pid}, _from, state) do
        {:reply, state.data |> Map.get(caller_pid) |> Enum.take(number_of_data) |> Enum.reverse(), state}
      end
    end
  end

  @callback add_data(data_sink_pid :: pid(), data_set_id :: term(), data :: Point.t() | [Point.t()]) :: :ok

  @callback get_data(data_sink_pid :: pid(), data_set_id :: term()) :: [Point.t()]

  @callback pop_data(data_sink_pid :: pid(), data_set_id :: term()) :: [Point.t()]

  @callback get_last_n_data(data_sink_pid :: pid(), data_set_id :: term(), n_points :: pos_integer()) :: [Point.t()]
end
