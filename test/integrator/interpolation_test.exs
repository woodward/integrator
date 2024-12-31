defmodule Integrator.InterpolationTest do
  @moduledoc false
  use Integrator.TestCase

  import Nx, only: :sigils

  alias Integrator.Interpolation

  describe "hermite_quartic" do
    setup do
      # These test values were obtained from Octave:
      # Generated using:
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset("AbsTol", 1.0e-14, "RelTol", 1.0e-14)
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0], opts);

      t = ~VEC[ 19.97226029930081   19.97424839002798 ]f64

      x = ~MAT[ 2.008585111348593e+00   2.008604708104012e+00
                1.188547490189183e-02   7.832739209072674e-03
            ]f64

      der = ~MAT[
        1.188547490189183e-02   1.107248473635211e-02   1.066709096215445e-02   8.641324907205110e-03   8.281808253394873e-03     7.832711009917654e-03   7.832739209072674e-03
       -2.044650564564792e+00  -2.042188551791212e+00  -2.040960496435665e+00  -2.034823361858414e+00  -2.033733968850626e+00    -2.032373024665618e+00  -2.032373099413282e+00
      ]f64

      t_out = ~VEC[ 19.97275732198261   19.97325434466440   19.97375136734619   19.97424839002798 ]f64

      expected_x_out = ~MAT[
       2.008590766279272e+00   2.008595916876415e+00   2.008600563898753e+00   2.008604708104012e+00
       1.087000165112446e-02   9.856055965852775e-03   8.843635825513105e-03   7.832739209072674e-03
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Interpolation.hermite_quartic(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-13, rtol: 1.0e-13)
      assert_nx_f64(x_out)
    end

    test "works for a single value of t (rather than an array of t)", %{t: t, x: x, der: der} do
      t_out = ~VEC[ 19.97375136734619 ]f64

      expected_x_out = ~MAT[
        2.008600563898753e+00
        8.843635825513105e-03
      ]f64

      x_out = Interpolation.hermite_quartic(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-14, rtol: 1.0e-14)
      assert_nx_f64(x_out)
    end
  end

  describe "hermite_cubic" do
    setup do
      # These test values were obtained from Octave:
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset("AbsTol", 1.0e-14, "RelTol", 1.0e-14, "Refine", 4)
      # [t,x] = ode23 (fvdp, [0, 20], [2, 0], opts);

      t = ~VEC[ 1.393975803797305   1.394019009661092 ]f64

      x = ~MAT[    1.155084908374247   1.155040344468921
                -1.031414373710718  -1.031449387283638
          ]f64

      der = ~MAT[
         -1.031414373710718  -1.031431879957980  -1.031440633890396  -1.031449387283638
         -0.810364414851849  -0.810389373959629  -0.810401854319440  -0.810414334531038
      ]f64

      t_out = ~VEC[ 1.393986605263252   1.393997406729198   1.394008208195145   1.394019009661092 ]f64

      expected_x_out = ~MAT[
          1.155073767539739   1.155062626610683   1.155051485587077   1.155040344468921
         -1.031423126901747  -1.031431880227575  -1.031440633688204  -1.031449387283638
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Interpolation.hermite_cubic(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-15, rtol: 1.0e-15)
      assert_nx_f64(x_out)
    end
  end
end
