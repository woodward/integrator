defmodule Integrator.DataSinkTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.DataSink

  test "can add and get data from a data sink" do
    {:ok, pid} = DataSink.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    data_set_id = self()

    DataSink.add_data(pid, data_set_id, data_point1)
    DataSink.add_data(pid, data_set_id, data_point2)

    points = DataSink.get_data(pid, data_set_id)

    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])
  end

  test "can add and get data from multiple data sources" do
    {:ok, pid} = DataSink.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)
    data_point3 = Nx.f32(3.0)
    data_point4 = Nx.f32(4.0)

    data_set_id = self()

    DataSink.add_data(pid, data_set_id, data_point1)
    DataSink.add_data(pid, data_set_id, data_point2)

    Task.async(fn ->
      data_set_id_2 = self()
      DataSink.add_data(pid, data_set_id_2, data_point3)
      DataSink.add_data(pid, data_set_id_2, data_point4)

      points = DataSink.get_data(pid, data_set_id_2)
      assert_nx_lists_equal(points, [Nx.f32(3.0), Nx.f32(4.0)])
    end)
    |> Task.await()

    points = DataSink.get_data(pid, data_set_id)

    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])
  end

  test "can add multiple data points in one call, and retrieve them from the data set" do
    {:ok, pid} = DataSink.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    data_set_id = self()

    DataSink.add_data(pid, data_set_id, [data_point1, data_point2])

    points = DataSink.get_data(pid, data_set_id)

    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])
  end

  test "can get n number of data points data set" do
    {:ok, pid} = DataSink.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)
    data_point3 = Nx.f32(3.0)
    data_point4 = Nx.f32(4.0)

    data_set_id = self()

    DataSink.add_data(pid, data_set_id, [data_point1, data_point2, data_point3, data_point4])

    points = DataSink.get_last_n_data(pid, data_set_id, 2)

    assert_nx_lists_equal(points, [Nx.f32(3.0), Nx.f32(4.0)])
  end

  test "pop_data returns and empties the data set" do
    {:ok, pid} = DataSink.start_link()

    data_point1 = Nx.f32(1.0)
    data_point2 = Nx.f32(2.0)

    data_set_id = self()

    DataSink.add_data(pid, data_set_id, [data_point1, data_point2])

    points = DataSink.pop_data(pid, data_set_id)
    assert_nx_lists_equal(points, [Nx.f32(1.0), Nx.f32(2.0)])

    points = DataSink.pop_data(pid, data_set_id)
    assert points == []
  end
end
