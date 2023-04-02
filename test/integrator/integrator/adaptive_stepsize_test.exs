defmodule Integrator.AdaptiveStepsizeTest do
  @moduledoc false
  use Integrator.TestCase

  describe "absolute_relative_norm/4" do
    test "general case" do
      x = [1.9754, -0.2665]
      x_old = [1.9957, -0.1232]
      absolute_tolerance = 1.0000e-06
      relative_tolerance = 1.0000e-03
      normcontrol = false
      y = [1.9754, -0.2665]

      expected_value = 4.7352e-03
    end
  end
end
