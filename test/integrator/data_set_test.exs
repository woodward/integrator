defmodule Integrator.DataSetTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.DataSet

  test "can add and get data from a data collector" do
    {:ok, pid} = DataSet.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    DataSet.add_data(pid, data_point1)
    DataSet.add_data(pid, data_point2)

    points = DataSet.get_data(pid)

    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])
  end

  test "can add multiple data points in one call, and retrieve them from the data set" do
    {:ok, pid} = DataSet.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    DataSet.add_data(pid, [data_point1, data_point2])

    points = DataSet.get_data(pid)

    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])
  end

  test "can get n number of data points data set" do
    {:ok, pid} = DataSet.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)
    data_point3 = Nx.f32(3.0)
    data_point4 = Nx.f32(4.0)

    DataSet.add_data(pid, [data_point1, data_point2, data_point3, data_point4])

    points = DataSet.get_last_n_data(pid, 2)

    assert_nx_lists_equal(points, [Nx.f32(3.0), Nx.f32(4.0)])
  end

  test "pop_data returns and empties the data set" do
    {:ok, pid} = DataSet.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    DataSet.add_data(pid, [data_point1, data_point2])

    points = DataSet.pop_data(pid)
    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])

    points = DataSet.pop_data(pid)
    assert points == []
  end
end
