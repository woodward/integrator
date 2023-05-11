defmodule IntegratorTest do
  @moduledoc false
  use Integrator.TestCase
  use Patch

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

    test "euler_equations" do
      t = Nx.tensor(0.1, type: :f32)
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f32)

      x_result = euler_equations(t, x)
      # From Octave:
      expected_x_result = Nx.tensor([6.0, -3.0, -1.02])
      assert_all_close(x_result, expected_x_result)
    end
  end

  describe "van_der_pol_fn" do
    setup do
      initial_x = Nx.tensor([2.0, 0.0], type: :f64)
      t_initial = Nx.tensor(0.0, type: :f64)
      t_final = Nx.tensor(20.0, type: :f64)

      [initial_x: initial_x, t_initial: t_initial, t_final: t_final]
    end

    test "performs the integration", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      opts = [
        type: :f64,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        norm_control: false,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      solution = Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(solution.output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - initial timestep specified", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset ("InitialStep", 0.1);
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0], opt);

      opts = [
        type: :f64,
        norm_control: false,
        initial_step: Nx.tensor(0.1, type: :f64),
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      solution = Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/initial_step_specified/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/initial_step_specified/x.csv")

      assert_nx_lists_equal(solution.output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - fixed output", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      t_range = Nx.linspace(t_initial, t_final, n: 21, type: :f64)

      opts = [
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        norm_control: false,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      solution = Integrator.integrate(&van_der_pol_fn/2, t_range, initial_x, opts)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/x.csv")

      assert_nx_lists_equal(solution.output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - high fidelity", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      opts = [
        abs_tol: Nx.tensor(1.0e-10, type: :f64),
        rel_tol: Nx.tensor(1.0e-10, type: :f64),
        integrator: :ode45,
        type: :f64,
        norm_control: false,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      solution = Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/x.csv")

      assert_nx_lists_equal(solution.output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - uses Bogacki-Shampine23", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      opts = [
        refine: 4,
        integrator: :ode23,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      solution = Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/x.csv")

      assert_nx_lists_equal(solution.output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "raises an exception for an undefined integrator", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      opts = [integrator: :undefined_integrator!]

      assert_raise RuntimeError, fn ->
        Integrator.integrate(&van_der_pol_fn/2, [t_initial, t_final], initial_x, opts)
      end
    end
  end

  describe "rigidode/euler_equations" do
    test "works with ode45" do
      # Octave:
      #   format long
      #   tspan = [0 12];
      #   x0 = [0; 1; 1];
      #   f_euler = @(t,x) [ x(2)*x(3) ; -x(1)*x(3) ; -0.51*x(1)*x(2) ];
      #   opt = odeset ("RelTol", 1.0e-07, "AbsTol", 1.0e-07);
      #   [t, x] = ode45(f_euler, tspan, x0, opt);

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(12.0, type: :f64)
      x0 = Nx.tensor([0.0, 1.0, 1.0], type: :f64)

      opts = [
        abs_tol: Nx.tensor(1.0e-07, type: :f64),
        rel_tol: Nx.tensor(1.0e-07, type: :f64),
        norm_control: false,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      solution = Integrator.integrate(&euler_equations/2, [t_start, t_end], x0, opts)

      assert solution.count_cycles__compute_step == 78
      assert solution.count_loop__increment_step == 78
      assert length(solution.ode_t) == 79
      assert length(solution.ode_x) == 79
      assert length(solution.output_t) == 313
      assert length(solution.output_x) == 313

      expected_t = read_nx_list("test/fixtures/octave_results/euler_equations/ode45/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/euler_equations/ode45/x.csv")

      assert_nx_lists_equal(solution.output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works with ode23" do
      # Octave:
      #   format long
      #   tspan = [0 12];
      #   x0 = [0; 1; 1];
      #   f_euler = @(t,x) [ x(2)*x(3) ; -x(1)*x(3) ; -0.51*x(1)*x(2) ];
      #   opt = odeset ("RelTol", 1.0e-07, "AbsTol", 1.0e-07);
      #   [t, x] = ode23(f_euler, tspan, x0, opt);

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(12.0, type: :f64)
      x0 = Nx.tensor([0.0, 1.0, 1.0], type: :f64)

      opts = [
        abs_tol: Nx.tensor(1.0e-07, type: :f64),
        rel_tol: Nx.tensor(1.0e-07, type: :f64),
        refine: 1,
        integrator: :ode23,
        type: :f64,
        norm_control: false,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      solution = Integrator.integrate(&euler_equations/2, [t_start, t_end], x0, opts)

      assert solution.count_cycles__compute_step == 847
      assert solution.count_loop__increment_step == 846
      assert length(solution.ode_t) == 847
      assert length(solution.ode_x) == 847
      assert length(solution.output_t) == 847
      assert length(solution.output_x) == 847

      expected_t = read_nx_list("test/fixtures/octave_results/euler_equations/ode23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/euler_equations/ode23/x.csv")

      assert_nx_lists_equal(solution.output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end
  end

  # ===========================================================================
  # Tests of private functions below here:

  describe "set_default_refine_opt/1" do
    setup do
      expose(Integrator, set_default_refine_opt: 1)
    end

    test "sets the default refine based on the integrator" do
      opts = [integrator: :ode45]

      assert private(Integrator.set_default_refine_opt(opts)) == [
               integrator: :ode45,
               refine: 4
             ]
    end

    test "uses the user-specified refine value if one is provided" do
      opts = [integrator: :ode45, refine: 3]

      assert private(Integrator.set_default_refine_opt(opts)) == [
               integrator: :ode45,
               refine: 3
             ]
    end

    test "uses the default :refine for :ode23" do
      opts = [integrator: :ode23]

      assert private(Integrator.set_default_refine_opt(opts)) == [
               integrator: :ode23,
               refine: 1
             ]
    end
  end

  describe "parse_start_end/1" do
    setup do
      expose(Integrator, parse_start_end: 1)
    end

    test "returns an array of times" do
      {t_start, t_end, fixed_times} = private(Integrator.parse_start_end([0.0, 1.0]))
      assert t_start == 0.0
      assert t_end == 1.0
      assert fixed_times == nil
    end

    test "creates an array of fixed_times if a single tensor is given" do
      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.5, type: :f64)
      t_values = Nx.linspace(t_start, t_end, n: 6, type: :f64)

      {t_start, t_end, fixed_times} = private(Integrator.parse_start_end(t_values))
      assert t_start.__struct__ == Nx.Tensor
      assert t_start == Nx.tensor(0.0, type: :f64)
      assert Nx.type(t_start) == {:f, 64}

      assert t_end.__struct__ == Nx.Tensor
      assert t_end == Nx.tensor(0.5, type: :f64)
      assert Nx.type(t_end) == {:f, 64}

      assert length(fixed_times) == 6
      [_first_time | [second_time | _rest]] = fixed_times
      assert second_time.__struct__ == Nx.Tensor
      assert second_time == Nx.tensor(0.1, type: :f64)
      assert Nx.type(second_time) == {:f, 64}
    end
  end
end
