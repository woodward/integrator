defmodule Integrator.DataCollectorTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.DataCollector

  test "can add and get data from a data collector" do
    {:ok, pid} = DataCollector.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    DataCollector.add_data(pid, data_point1)
    DataCollector.add_data(pid, data_point2)

    points = DataCollector.get_data(pid)

    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])
  end

  test "can add multiple data points in one call, and retrieve them from the data collector" do
    {:ok, pid} = DataCollector.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    DataCollector.add_data(pid, [data_point1, data_point2])

    points = DataCollector.get_data(pid)

    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])
  end

  test "can get n number of data points data collector" do
    {:ok, pid} = DataCollector.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)
    data_point3 = Nx.f32(3.0)
    data_point4 = Nx.f32(4.0)

    DataCollector.add_data(pid, [data_point1, data_point2, data_point3, data_point4])

    points = DataCollector.get_last_n_data(pid, 2)

    assert_nx_lists_equal(points, [Nx.f32(3.0), Nx.f32(4.0)])
  end
end
