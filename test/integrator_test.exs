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

  describe "rigidode/euler_equations" do
    test "works with ode45" do
      # Octave:
      #   format long
      #   tspan = [0 12];
      #   x0 = [0; 1; 1];
      #   f_euler = @(t,x) [ x(2)*x(3) ; -x(1)*x(3) ; -0.51*x(1)*x(2) ];
      #   opt = odeset ("RelTol", 1.0e-07, "AbsTol", 1.0e-07);
      #   [t, x] = ode45(f_euler, tspan, x0, opt);

      t_start = 0.0
      t_end = 12.0
      x0 = Nx.tensor([0.0, 1.0, 1.0], type: :f64)
      opts = [abs_tol: 1.0e-07, rel_tol: 1.0e-07]

      solution = Integrator.integrate(&euler_equations/2, [t_start, t_end], x0, opts)

      assert solution.count_cycles__compute_step == 78
      assert solution.count_loop__increment_step == 78
      assert length(solution.ode_t) == 79
      assert length(solution.ode_x) == 79
      assert length(solution.output_t) == 313
      assert length(solution.output_x) == 313

      expected_t = read_csv("test/fixtures/octave_results/euler_equations/ode45/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/euler_equations/ode45/x.csv")

      # data = solution.output_t |> Enum.join("\n")
      # File.write!("test/fixtures/octave_results/euler_equations/ode45/t_elixir.csv", data)

      # data =
      #   solution.output_x
      #   |> Enum.map(fn xn -> "#{Nx.to_number(xn[0])}  #{Nx.to_number(xn[1])}  #{Nx.to_number(xn[2])}  " end)
      #   |> Enum.join("\n")

      # File.write!("test/fixtures/octave_results/euler_equations/ode45/x_elixir.csv", data)

      assert_lists_equal(solution.output_t, expected_t, 1.0e-05)
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

      t_start = 0.0
      t_end = 12.0
      x0 = Nx.tensor([0.0, 1.0, 1.0], type: :f64)
      opts = [abs_tol: 1.0e-07, rel_tol: 1.0e-07, refine: 1, integrator: :ode23]

      solution = Integrator.integrate(&euler_equations/2, [t_start, t_end], x0, opts)

      assert solution.count_cycles__compute_step == 847
      assert solution.count_loop__increment_step == 846
      assert length(solution.ode_t) == 847
      assert length(solution.ode_x) == 847
      assert length(solution.output_t) == 847
      assert length(solution.output_x) == 847

      expected_t = read_csv("test/fixtures/octave_results/euler_equations/ode23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/euler_equations/ode23/x.csv")

      # data = solution.output_t |> Enum.join("\n")
      # File.write!("test/fixtures/octave_results/euler_equations/ode23/t_elixir.csv", data)

      # data =
      #   solution.output_x
      #   |> Enum.map(fn xn -> "#{Nx.to_number(xn[0])}  #{Nx.to_number(xn[1])}  #{Nx.to_number(xn[2])}  " end)
      #   |> Enum.join("\n")

      # File.write!("test/fixtures/octave_results/euler_equations/ode23/x_elixir.csv", data)

      assert_lists_equal(solution.output_t, expected_t, 1.0e-05)
      assert_nx_lists_equal(solution.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end
  end

  describe "merge_default_opts/1" do
    setup do
      expose(Integrator, merge_default_opts: 1)
    end

    test "has defaults for Integrator and Utils" do
      opts = []

      assert private(Integrator.merge_default_opts(opts)) == [
               integrator: :ode45,
               abs_tol: 1.0e-06,
               rel_tol: 1.0e-03,
               norm_control: true,
               refine: 4
             ]
    end

    test "the default :refine is changed to 1 if integrator: :ode23" do
      opts = [integrator: :ode23]

      assert private(Integrator.merge_default_opts(opts)) == [
               abs_tol: 1.0e-06,
               rel_tol: 1.0e-03,
               norm_control: true,
               integrator: :ode23,
               refine: 1
             ]
    end

    test "allows all default values to be overridden" do
      opts = [
        abs_tol: 1.0e-12,
        rel_tol: 1.0e-13,
        norm_control: false,
        integrator: :ode23,
        refine: 3
      ]

      assert private(Integrator.merge_default_opts(opts)) == [
               abs_tol: 1.0e-12,
               rel_tol: 1.0e-13,
               norm_control: false,
               integrator: :ode23,
               refine: 3
             ]
    end
  end
end
