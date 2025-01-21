defmodule Integrator.PointTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.Point

  describe "split_points_into_t_and_x" do
    test "creates a list of t and x from a list of points" do
      points = [
        %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])},
        %Point{t: Nx.f64(0.2), x: Nx.f64([3.0, 4.0])},
        %Point{t: Nx.f64(0.3), x: Nx.f64([5.0, 6.0])},
        %Point{t: Nx.f64(0.4), x: Nx.f64([7.0, 8.0])}
      ]

      {t, x} = Point.split_points_into_t_and_x(points)

      expected_t = [
        Nx.f64(0.1),
        Nx.f64(0.2),
        Nx.f64(0.3),
        Nx.f64(0.4)
      ]

      expected_x = [
        Nx.f64([1.0, 2.0]),
        Nx.f64([3.0, 4.0]),
        Nx.f64([5.0, 6.0]),
        Nx.f64([7.0, 8.0])
      ]

      assert t == expected_t
      assert x == expected_x
    end
  end

  describe "to_number/1" do
    test "converts a point from Nx tensors to numbers" do
      point = %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])}
      assert Point.to_number(point) == %Point{t: 0.1, x: [1.0, 2.0]}
    end
  end

  describe "filter_out_points_with_same_t/1" do
    test "gets rid of points with the same t value" do
      points = [
        %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])},
        %Point{t: Nx.f64(0.2), x: Nx.f64([3.0, 4.0])},
        %Point{t: Nx.f64(0.2), x: Nx.f64([5.0, 6.0])},
        %Point{t: Nx.f64(0.3), x: Nx.f64([7.0, 8.0])}
      ]

      filtered_points = Point.filter_out_points_with_same_t(points)

      assert filtered_points == [
               %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])},
               %Point{t: Nx.f64(0.2), x: Nx.f64([3.0, 4.0])},
               %Point{t: Nx.f64(0.3), x: Nx.f64([7.0, 8.0])}
             ]
    end
  end
end
