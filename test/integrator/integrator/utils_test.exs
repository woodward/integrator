defmodule Integrator.UtilsTest do
  @moduledoc false
  use Integrator.TestCase

  alias Integrator.Utils

  describe "absolute_relative_norm/6" do
    test "general case" do
      # Values from Octave:
      #
      # x = [1.97537683003, -0.26652885197];
      # x_old = [1.99566026409, -0.12317664679];
      # AbsTol = 1.0000e-06;
      # RelTol = 1.0000e-03;
      # normcontrol = false;
      # y = [1.97537723429, -0.26653011403];
      #
      # AbsRel_norm (x, x_old, AbsTol, RelTol, normcontrol, y)

      x = Nx.tensor([1.97537683003, -0.26652885197])
      x_old = Nx.tensor([1.99566026409, -0.12317664679])
      absolute_tolerance = 1.0000e-06
      relative_tolerance = 1.0000e-03
      normcontrol = false
      y = Nx.tensor([1.97537723429, -0.26653011403])
      expected_norm = Nx.tensor(0.00473516383083)

      norm = Utils.absolute_relative_norm(x, x_old, y, absolute_tolerance, relative_tolerance, normcontrol: normcontrol)

      assert_all_close(norm, expected_norm, atol: 1.0e-04, rtol: 1.0e-04)
    end
  end
end
