defmodule Integrator.UtilsTest do
  @moduledoc false
  use Integrator.TestCase
  import Nx, only: :sigils

  alias Integrator.Utils

  describe "abs_rel_norm/6" do
    test "general case for norm_control: false" do
      # These test values were obtained from Octave:
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
      # These test values were obtained from Octave:
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

  describe "hermite_quartic_interpolation" do
    setup do
      # These test values were obtained from Octave:

      t = ~V[ 19.4067624192  19.7106968201 ]f64

      x = ~M[ 1.49652504841   1.92651431954
              2.10676183153   0.73529371547
            ]f64

      der = ~M[
           2.10676183153   1.85704689071   1.69384014677   1.09328301986   1.20767740745   1.06405291882   0.73529371547
          -4.10804009148  -4.66882299445  -4.71039008294  -4.47781878341  -4.51898062008  -4.30261858646  -3.92023192271
      ]f64

      t_out = ~V[ 19.4827460194  19.5587296196  19.6347132198  19.7106968201 ]f64

      expected_x_out = ~M[
        1.64398703647   1.76488148020   1.85862355568   1.92651431954
        1.77097584066   1.41075566029   1.05684789008   0.73529371547
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Utils.hermite_quartic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-9, rtol: 1.0e-9)
    end
  end
end
