defmodule Integrator.DataCollectorTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.DataCollector

  test "can add and get data from a data collector" do
    {:ok, pid} = DataCollector.start_link()

    data_point1 = Nx.tensor(1.0, type: :f32)
    data_point2 = Nx.tensor(2.0, type: :f32)

    DataCollector.add_data(pid, data_point1)
    DataCollector.add_data(pid, data_point2)

    points = DataCollector.get_data(pid)

    assert points == [Nx.tensor(1.0, type: :f32), Nx.tensor(2.0, type: :f32)]
  end

  test "can add multiple data points in one call, and retrieve them from the data collector" do
    {:ok, pid} = DataCollector.start_link()

    data_point1 = Nx.tensor(1.0, type: :f32)
    data_point2 = Nx.tensor(2.0, type: :f32)

    DataCollector.add_data(pid, [data_point1, data_point2])

    points = DataCollector.get_data(pid)

    assert points == [Nx.tensor(1.0, type: :f32), Nx.tensor(2.0, type: :f32)]
  end
end
