defmodule Integrator.SampleEqnsTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  import Nx, only: :sigils

  alias Integrator.SampleEqns

  describe "van_der_pol_fn/2" do
    test "returns the correct values to the right precision" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   t = 0.3;  ## Not used
      #   x = [0.25, 1.75]
      #   fvdp(t, x)

      t = ~VEC[  0.3  ]f64
      x = ~VEC[  0.25  1.75  ]f64

      result = SampleEqns.van_der_pol_fn(t, x)

      # Expected value from Octave:
      expected_result = ~VEC[  1.750000000000000   1.390625000000000  ]f64
      assert_all_close(result, expected_result, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  describe "euler_equations/2" do
    test "returns the correct values to the right precision" do
      # Octave:
      #   format long
      #   x = [0.5; 1.2; 2.5];
      #   f_euler = @(t,x) [ x(2)*x(3) ; -x(1)*x(3) ; -0.51*x(1)*x(2) ];
      #   t = 0.3;  ## Not used
      #   f_euler(t, x)
      t = ~VEC[  0.3  ]f64
      x = ~VEC[  0.5  1.2  2.5  ]f64

      x_result = SampleEqns.euler_equations(t, x)

      # Expected value from Octave:
      expected_x_result = ~VEC[   3.000000000000000  -1.250000000000000  -0.306000000000000  ]f64
      assert_all_close(x_result, expected_x_result, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  describe "falling_particle/2" do
    test "returns the correct values to the right precision" do
      # Octave:
      #   format long
      #   x = [0.5; 1.2];
      #   dydt = @(t,x) [x(2); -9.81];
      #   t = 0.3;  ## Not used
      #   dydt(t, x)
      t = ~VEC[  0.3  ]f64
      x = ~VEC[  0.5  1.2  ]f64

      x_result = SampleEqns.falling_particle(t, x)

      # Expected value from Octave:
      expected_x_result = ~VEC[  1.200000000000000  -9.810000000000000  ]f64
      assert_all_close(x_result, expected_x_result, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end
end
