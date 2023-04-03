defmodule Integrator.UtilsTest do
  @moduledoc false
  use Integrator.TestCase

  alias Integrator.Utils

  describe "abs_rel_norm/6" do
    test "general case for norm_control: false" do
      # Values from Octave:
      #
      # x = [1.97537683003, -0.26652885197]
      # x_old = [1.99566026409, -0.12317664679]
      # AbsTol = 1.0000e-06
      # RelTol = 1.0000e-03
      # normcontrol = false
      # y = [1.97537723429, -0.26653011403]
      #
      # AbsRel_norm (x, x_old, AbsTol, RelTol, norm_control, y)

      x = Nx.tensor([1.97537683003, -0.26652885197])
      x_old = Nx.tensor([1.99566026409, -0.12317664679])
      abs_tolerance = 1.0000e-06
      rel_tolerance = 1.0000e-03
      y = Nx.tensor([1.97537723429, -0.26653011403])
      expected_norm = Nx.tensor(0.00473516383083)

      norm = Utils.abs_rel_norm(x, x_old, y, abs_tolerance, rel_tolerance, norm_control: false)

      assert_all_close(norm, expected_norm, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "general case for norm_control: true" do
      # Values from Octave:
      #
      # x = [1.99465419035, 0.33300240425]
      # x_old = [1.64842646336, 1.78609260054]
      # AbsTol = 1.00000000000e-06
      # RelTol = 0.00100000000000
      # normcontrol = true
      # y = [1.99402286380, 0.33477644992]
      #
      # AbsRel_norm (x, x_old, AbsTol, RelTol, norm_control, y)

      x = Nx.tensor([1.99465419035, 0.33300240425])
      x_old = Nx.tensor([1.64842646336, 1.78609260054])
      abs_tolerance = 1.0000e-06
      rel_tolerance = 1.0000e-03
      y = Nx.tensor([1.99402286380, 0.33477644992])
      expected_norm = Nx.tensor(0.77474409123)

      norm = Utils.abs_rel_norm(x, x_old, y, abs_tolerance, rel_tolerance, norm_control: true)

      assert_all_close(norm, expected_norm, atol: 1.0e-04, rtol: 1.0e-04)
    end
  end
end
