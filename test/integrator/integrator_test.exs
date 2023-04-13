defmodule IntegratorTest do
  @moduledoc false
  use Integrator.TestCase

  describe "test setup" do
    test "van_der_pol_fn" do
      x = Nx.tensor(0.0341, type: :f32)
      y = Nx.tensor([1.9975, -0.0947], type: :f32)

      # From octave with y(1) -> y(0) and y(2) -> y(1):
      # fvdp = @(t,y) [y(1); (1 - y(0)^2) * y(1) - y(0)];
      y0 = 1.9975
      y1 = -0.0947
      expected_y0 = y1
      expected_y1 = (1.0 - y0 * y0) * y1 - y0
      assert expected_y0 == -0.0947
      assert expected_y1 == -1.714346408125

      y_result = van_der_pol_fn(x, y)
      expected_y_result = Nx.tensor([expected_y0, expected_y1])
      assert_all_close(y_result, expected_y_result)
    end
  end

  describe "overall" do
    setup do
      initial_y = Nx.tensor([2.0, 0.0])
      t_initial = 0.0
      t_final = 20.0

      [initial_y: initial_y, t_initial: t_initial, t_final: t_final]
    end

    test "performs the integration", %{initial_y: initial_y, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave/function/ode45.html
      #
      # fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
      # [t,y] = ode45 (fvdp, [0, 20], [2, 0]);

      solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_y)

      expected_t = read_csv("test/fixtures/integrator/integrator/runge_kutta_45_test/time.csv")
      expected_y = read_nx_list("test/fixtures/integrator/integrator/runge_kutta_45_test/x.csv")

      assert_lists_equal(solution.output_t, expected_t, 0.01)
      assert_nx_lists_equal(solution.output_x, expected_y, atol: 0.1, rtol: 0.1)
    end

    test "performs the integration - high fidelity", %{initial_y: initial_y, t_initial: t_initial, t_final: t_final} do
      opts = [abs_tol: 1.0e-10, rel_tol: 1.0e-10]
      solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_y, opts)

      expected_t = read_csv("test/fixtures/integrator/integrator/runge_kutta_45_test/time_high_fidelity.csv")
      expected_y = read_nx_list("test/fixtures/integrator/integrator/runge_kutta_45_test/x_high_fidelity.csv")

      assert_lists_equal(solution.output_t, expected_t, 1.0e-02)
      assert_nx_lists_equal(solution.output_x, expected_y, atol: 1.0e-02, rtol: 1.0e-02)
    end
  end
end
