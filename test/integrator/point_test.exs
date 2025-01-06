defmodule Integrator.PointTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  import Nx, only: :sigils

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

  describe "new/2" do
    test "creates a list of Points from t and x values - 1 point" do
      t = ~VEC[0.1]f64
      x = ~MAT[
           1.0
           2.0
      ]f64

      points = Point.convert_to_points(t, x)

      {point1} = points
      assert point1 == %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])}
    end

    test "creates a list of Points from t and x values - 2 points" do
      t = ~VEC[0.1 0.2]f64
      x = ~MAT[
           1.0   3.0
           2.0   4.0
      ]f64

      points = Point.convert_to_points(t, x)

      {point1, point2} = points

      assert point1 == %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])}
      assert point2 == %Point{t: Nx.f64(0.2), x: Nx.f64([3.0, 4.0])}
    end
  end

  test "creates a list of Points from t and x values - 3 points" do
    t = ~VEC[0.1 0.2 0.3]f64
    x = ~MAT[
           1.0   3.0   5.0
           2.0   4.0   6.0
      ]f64

    points = Point.convert_to_points(t, x)

    {point1, point2, point3} = points

    assert point1 == %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])}
    assert point2 == %Point{t: Nx.f64(0.2), x: Nx.f64([3.0, 4.0])}
    assert point3 == %Point{t: Nx.f64(0.3), x: Nx.f64([5.0, 6.0])}
  end

  test "creates a list of Points from t and x values - 4 points" do
    t = ~VEC[0.1 0.2 0.3 0.4]f64
    x = ~MAT[
           1.0   3.0  5.0  7.0
           2.0   4.0  6.0  8.0
      ]f64

    points = Point.convert_to_points(t, x)

    {point1, point2, point3, point4} = points

    assert point1 == %Point{t: Nx.f64(0.1), x: Nx.f64([1.0, 2.0])}
    assert point2 == %Point{t: Nx.f64(0.2), x: Nx.f64([3.0, 4.0])}
    assert point3 == %Point{t: Nx.f64(0.3), x: Nx.f64([5.0, 6.0])}
    assert point4 == %Point{t: Nx.f64(0.4), x: Nx.f64([7.0, 8.0])}
  end

  test "raises an exception if there are 5 points" do
    t = ~VEC[0.1 0.2 0.3 0.4 0.5]f64
    x = ~MAT[
           1.0   3.0  5.0  7.0  9.0
           2.0   4.0  6.0  8.0 10.0
      ]f64

    assert_raise RuntimeError, fn ->
      Point.convert_to_points(t, x)
    end
  end
end
