defmodule Integrator.DummyOutput do
  @moduledoc """
  Shows how to use an output function
  """

  use GenServer

  defstruct x_data: [],
            t_data: []

  def start_link(opts) do
    GenServer.start_link(__MODULE__, :ok, opts)
  end

  def add_data(name, data) do
    GenServer.call(name, {:add_data, data})
  end

  def add_data_and_halt(name, data) do
    GenServer.call(name, {:add_data_and_halt, data})
  end

  def get_x(name) do
    GenServer.call(name, :get_x)
  end

  def get_t(name) do
    GenServer.call(name, :get_t)
  end

  @impl true
  def init(:ok) do
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call({:add_data, %{x: new_x, t: new_t}}, _from, state) when is_list(new_x) and is_list(new_t) do
    {:reply, :continue, add_new_data_to_state(state, new_x, new_t)}
  end

  @impl true
  def handle_call({:add_data, %{x: new_x, t: new_t}}, _from, state) do
    {:reply, :continue, add_new_data_to_state(state, [new_x], [new_t])}
  end

  @impl true
  def handle_call({:add_data_and_halt, %{x: new_x, t: new_t}}, _from, state) when is_list(new_x) and is_list(new_t) do
    {:reply, :halt, add_new_data_to_state(state, new_x, new_t)}
  end

  @impl true
  def handle_call({:add_data_and_halt, %{x: new_x, t: new_t}}, _from, state) do
    {:reply, :halt, add_new_data_to_state(state, [new_x], [new_t])}
  end

  @impl true
  def handle_call(:get_x, _from, state) do
    {:reply, state.x_data |> Enum.reverse(), state}
  end

  @impl true
  def handle_call(:get_t, _from, state) do
    {:reply, state.t_data |> Enum.reverse(), state}
  end

  defp add_new_data_to_state(state, new_x, new_t) do
    x = new_x |> Enum.reduce(state.x_data, fn x, acc -> [x | acc] end)
    t = new_t |> Enum.reduce(state.t_data, fn t, acc -> [t | acc] end)
    %{state | x_data: x, t_data: t}
  end
end
