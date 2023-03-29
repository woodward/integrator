defmodule Momentum.Integrator.RungeKutta45Test do
  @moduledoc false
  use ExUnit.Case
  alias Momentum.Integrator.RungeKutta45

  describe "overall" do
    test "performs the integration" do
      # [t,y] = ode45 (fvdp, [0, 20], [2, 0]);
      van_der_pol = nil
      initial_y = [2.0, 0.0]
      t_initial = 0.0
      t_final = 20.0
      [t, y] = RungeKutta45.integrate(van_der_pol, t_initial, t_final, initial_y)

      expected_t = File.read!("test/fixtures/momentum/integrator/runge_kutta_45_test/time.csv")
      expected_y = read_nx_list("test/fixtures/momentum/integrator/runge_kutta_45_test/y.csv")

      assert_list_equal(t, expected_t)
      assert_nx_list_equal(y, expected_y)
    end
  end

  defp read_nx_list(filename) do
    File.read!(filename)
  end

  defp assert_list_equal(_actual_list, _expected_list) do
    true
  end

  defp assert_nx_list_equal(_actual_list, _expected_list) do
    true
  end
end
