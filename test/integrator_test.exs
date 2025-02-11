defmodule IntegratorTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.RungeKutta.BogackiShampine23
  alias Integrator.RungeKutta.DormandPrince45
  alias Integrator.DataSink
  alias Integrator.Point

  describe "van_der_pol_fn" do
    setup do
      initial_x = Nx.f64([2.0, 0.0])
      t_initial = Nx.f64(0.0)
      t_final = Nx.f64(20.0)

      [initial_x: initial_x, t_initial: t_initial, t_final: t_final]
    end

    test "performs the integration", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(pid, &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      _solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)

      {output_t, output_x} = DataSink.get_data(pid) |> Point.split_points_into_t_and_x()
      assert length(output_t) == 201
      assert length(output_x) == 201

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - initial timestep specified", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset ("InitialStep", 0.1);
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0], opt);

      {:ok, pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(pid, &1)

      opts = [
        type: :f64,
        norm_control?: false,
        initial_step: Nx.f64(0.1),
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      _solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      {output_t, output_x} = DataSink.get_data(pid) |> Point.split_points_into_t_and_x()

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/initial_step_specified/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/initial_step_specified/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - fixed output", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(pid, &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn,
        fixed_output_times?: true,
        fixed_output_step: 1.0
      ]

      _solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      {output_t, output_x} = DataSink.get_data(pid) |> Point.split_points_into_t_and_x()

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration - high fidelity", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      {:ok, pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(pid, &1)

      opts = [
        abs_tol: Nx.f64(1.0e-10),
        rel_tol: Nx.f64(1.0e-10),
        integrator: DormandPrince45,
        type: :f64,
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      _solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      {output_t, output_x} = DataSink.get_data(pid) |> Point.split_points_into_t_and_x()

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - uses Bogacki-Shampine23", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      {:ok, pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(pid, &1)

      opts = [
        type: :f64,
        refine: 4,
        integrator: BogackiShampine23,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      _solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      {output_t, output_x} = DataSink.get_data(pid) |> Point.split_points_into_t_and_x()

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "raises an exception for an undefined integrator", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      opts = [integrator: :undefined_integrator!]

      assert_raise NimbleOptions.ValidationError, fn ->
        Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
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
      {:ok, pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(12.0)
      x0 = Nx.f64([0.0, 1.0, 1.0])

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-07),
        rel_tol: Nx.f64(1.0e-07),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      solution = Integrator.integrate(&euler_equations/2, t_start, t_end, x0, opts)
      {output_t, output_x} = DataSink.get_data(pid) |> Point.split_points_into_t_and_x()

      assert_nx_equal(solution.count_cycles__compute_step, Nx.s32(78))
      assert_nx_equal(solution.count_loop__increment_step, Nx.s32(78))
      # assert length(solution.ode_t) == 79
      # assert length(solution.ode_x) == 79
      assert length(output_t) == 313
      assert length(output_x) == 313

      expected_t = read_nx_list("test/fixtures/octave_results/euler_equations/ode45/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/euler_equations/ode45/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works with ode23" do
      {:ok, pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(pid, &1)

      # Octave:
      #   format long
      #   tspan = [0 12];
      #   x0 = [0; 1; 1];
      #   f_euler = @(t,x) [ x(2)*x(3) ; -x(1)*x(3) ; -0.51*x(1)*x(2) ];
      #   opt = odeset ("RelTol", 1.0e-07, "AbsTol", 1.0e-07);
      #   [t, x] = ode23(f_euler, tspan, x0, opt);

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(12.0)
      x0 = Nx.f64([0.0, 1.0, 1.0])

      opts = [
        abs_tol: Nx.f64(1.0e-07),
        rel_tol: Nx.f64(1.0e-07),
        refine: 1,
        integrator: BogackiShampine23,
        type: :f64,
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      solution = Integrator.integrate(&euler_equations/2, t_start, t_end, x0, opts)
      {output_t, output_x} = DataSink.get_data(pid) |> Point.split_points_into_t_and_x()

      assert_nx_equal(solution.count_cycles__compute_step, Nx.s32(847))
      assert_nx_equal(solution.count_loop__increment_step, Nx.s32(846))
      # assert length(solution.ode_t) == 847
      # assert length(solution.ode_x) == 847
      assert length(output_t) == 847
      assert length(output_x) == 847

      expected_t = read_nx_list("test/fixtures/octave_results/euler_equations/ode23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/euler_equations/ode23/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end
  end
end
