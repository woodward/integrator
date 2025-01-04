defmodule Integrator.PointTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  import Nx, only: :sigils

  alias Integrator.Point

  describe "new/2" do
    test "creates a list of Points from t and x values" do
      t = ~VEC[0.1 0.2]f64
      x = ~MAT[
           1.0   2.0
           3.0   4.0
      ]f64

      points = Point.points_from_t_and_x(t, x)

      [point1, point2] = points

      assert point1 == %Point{t: 0.1, x: [1.0, 3.0]}
      assert point2 == %Point{t: 0.2, x: [2.0, 4.0]}
    end
  end

  describe "try to convert it to nx" do
    test "what does this do?" do
      x = ~MAT[
        1.0   2.0
        3.0   4.0
      ]f64

      {_x0, _x1} = Point.what_does_this_do?(x)
      # dbg(x0)
      # dbg(x1)
    end
  end
end
