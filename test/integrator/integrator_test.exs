defmodule IntegratorTest do
  @moduledoc false
  use Integrator.TestCase

  describe "test setup" do
    test "van_der_pol_fn" do
      t = Nx.tensor(0.0341, type: :f32)
      x = Nx.tensor([1.9975, -0.0947], type: :f32)

      # From octave with x(1) -> x(0) and x(2) -> x(1):
      # fvdp = @(t,x) [x(1); (1 - x(0)^2) * x(1) - x(0)];
      x0 = 1.9975
      x1 = -0.0947
      expected_x0 = x1
      expected_x1 = (1.0 - x0 * x0) * x1 - x0
      assert expected_x0 == -0.0947
      assert expected_x1 == -1.714346408125

      x_result = van_der_pol_fn(t, x)
      expected_x_result = Nx.tensor([expected_x0, expected_x1])
      assert_all_close(x_result, expected_x_result)
    end
  end

  describe "overall" do
    setup do
      initial_x = Nx.tensor([2.0, 0.0])
      t_initial = 0.0
      t_final = 20.0

      [initial_x: initial_x, t_initial: t_initial, t_final: t_final]
    end

    test "performs the integration", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      solution = Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x)

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_lists_equal(solution.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - fixed output", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      t_range = Nx.linspace(t_initial, t_final, n: 21, type: :f64)
      solution = Integrator.integrate(&van_der_pol_fn/2, t_range, initial_x)

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/x.csv")

      assert_lists_equal(solution.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - high fidelity", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      opts = [abs_tol: 1.0e-10, rel_tol: 1.0e-10, integrator: :ode45]
      solution = Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/x.csv")

      assert_lists_equal(solution.output_t, expected_t, 1.0e-05)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - uses Bogacki-Shampine23", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      opts = [refine: 4, integrator: :ode23]

      solution = Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/x.csv")

      assert_lists_equal(solution.output_t, expected_t, 1.0e-05)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "raises an exception for an undefined integrator", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      opts = [integrator: :undefined_integrator!]

      assert_raise RuntimeError, fn ->
        Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)
      end
    end
  end
end
