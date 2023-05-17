defmodule Integrator.DemoTest do
  @moduledoc false
  use Integrator.TestCase
  import Nx, only: :sigils
  alias Integrator.Demo

  describe "van_der_pol_fn/2" do
    test "returns the correct values to the right precision" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   t = 0.3;
      #   x = [0.25, 1.75]
      #   fvdp(t, x)

      t = ~V[  0.3  ]f64
      x = ~V[  0.25  1.75  ]f64

      result = Demo.van_der_pol_fn(t, x)

      # Expected value from Octave:
      expected_result = ~V[  1.750000000000000   1.390625000000000  ]f64
      assert_all_close(result, expected_result, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end
end
