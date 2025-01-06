defmodule Integrator.AdaptiveStepsizeRefactorTest do
  @moduledoc false
  use Integrator.TestCase, async: true
  import Nx, only: :sigils

  alias Integrator.AdaptiveStepsizeRefactor
  alias Integrator.AdaptiveStepsizeRefactor.NxOptions
  alias Integrator.DataCollector
  alias Integrator.ExternalFnAdapter
  alias Integrator.NonLinearEqnRoot
  alias Integrator.RungeKutta.DormandPrince45
  alias Integrator.SampleEqns

  describe "integrate" do
    @tag :skip
    test "works - no data interpolation (refine == 1)" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      ode_fn = &SampleEqns.van_der_pol_fn/2
      order = DormandPrince45.order()

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        refine: 1,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      assert result.count_cycles__compute_step == Nx.s32(78)
      assert result.count_loop__increment_step == Nx.s32(50)
      # assert length(result.ode_t) == 51
      # assert length(result.ode_x) == 51

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/x.csv")

      all_points = DataCollector.all_data(pid)
      output_t = all_points |> Enum.map(& &1.t)
      output_x = all_points |> Enum.map(& &1.x)
      assert length(output_t) == 51
      assert length(output_x) == 51

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      assert result.overall_elapsed_time_μs(result) > 1
      assert result.step_elapsed_time_μs(result) > 1
    end
  end

  describe "starting_stepsize" do
    test "works" do
      order = 5
      t0 = 0.0
      x0 = ~VEC[2.0 0.0]f64
      abs_tol = 1.0e-06
      rel_tol = 1.0e-03
      norm_control? = Nx.u8(0)

      starting_stepsize =
        AdaptiveStepsizeRefactor.starting_stepsize(order, &van_der_pol_fn/2, t0, x0, abs_tol, rel_tol, norm_control?)

      assert_all_close(starting_stepsize, Nx.tensor(0.068129, type: :f64), atol: 1.0e-6, rtol: 1.0e-6)
    end

    test "works - high fidelity ballode example to double precision accuracy (works!!!)" do
      order = 5
      t0 = ~VEC[  0.0  ]f64
      x0 = ~VEC[  0.0 20.0  ]f64
      abs_tol = Nx.tensor(1.0e-14, type: :f64)
      rel_tol = Nx.tensor(1.0e-14, type: :f64)
      norm_control? = Nx.u8(0)
      ode_fn = &SampleEqns.falling_particle/2

      starting_stepsize = AdaptiveStepsizeRefactor.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, norm_control?)
      assert_all_close(starting_stepsize, Nx.tensor(0.001472499532027109, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "does NOT work for precision :f16" do
      order = 5
      t0 = ~VEC[  0.0  ]f16
      x0 = ~VEC[  2.0  0.0  ]f16
      abs_tol = Nx.tensor(1.0e-06, type: :f16)
      rel_tol = Nx.tensor(1.0e-03, type: :f16)
      norm_control? = Nx.u8(0)
      ode_fn = &SampleEqns.van_der_pol_fn/2

      starting_stepsize = AdaptiveStepsizeRefactor.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, norm_control?)

      zero_stepsize_which_is_bad = Nx.tensor(0.0, type: :f16)

      assert_all_close(starting_stepsize, zero_stepsize_which_is_bad, atol: 1.0e-14, rtol: 1.0e-14)

      # The starting_stepsize is zero because d2 goes to infinity:
      # abs_rel_norm = abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, opts)
      # Values for abs_rel_norm and h0 captured from Elixir output:
      abs_rel_norm = Nx.tensor(999.5, type: :f16)
      h0 = Nx.tensor(0.01000213623046875, type: :f16)
      one = Nx.tensor(1, type: :f16)

      #  d2 = one / h0 * abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, opts)
      d2 = Nx.divide(one, h0) |> Nx.multiply(abs_rel_norm)
      # d2 being infinity causes the starting_stepsize to be zero:
      assert_all_close(d2, Nx.Constants.infinity(), atol: 1.0e-14, rtol: 1.0e-14)
    end
  end

  describe "convert_to_nx_options" do
    test "uses the defaults from nimble options (and defaults for machine_eps and tolerance in the type specified)" do
      use_default_opts = []
      t_start = 0.0
      t_end = 10.0
      order = 5

      nx_options = AdaptiveStepsizeRefactor.convert_to_nx_options(t_start, t_end, order, use_default_opts)
      assert %NxOptions{} = nx_options

      assert nx_options.type == {:f, 32}
      assert nx_options.max_number_of_errors == Nx.s32(5_000)
      assert nx_options.dt_max == Nx.f32(1.0)
      assert nx_options.refine == 4
      assert nx_options.speed == Nx.Constants.infinity(:f32)
      assert nx_options.fixed_output_times? == Nx.u8(0)
      assert nx_options.fixed_output_dt == Nx.f32(0.0)
      assert nx_options.order == 5
      assert nx_options.norm_control? == Nx.u8(1)
      assert nx_options.abs_tol == Nx.f32(1.0e-06)
      assert nx_options.rel_tol == Nx.f32(1.0e-03)

      assert nx_options.event_fn_adapter == %ExternalFnAdapter{}
      assert nx_options.event_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)

      assert nx_options.output_fn_adapter == %ExternalFnAdapter{}
      assert nx_options.output_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)

      assert nx_options.zero_fn_adapter == %ExternalFnAdapter{}
      assert nx_options.zero_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)

      # --------------------------------------
      # Values are passed on to NonLinearEqnRoot.NxOptions:
      non_linear_eqn_root_nx_options = nx_options.non_linear_eqn_root_nx_options
      assert %NonLinearEqnRoot.NxOptions{} = non_linear_eqn_root_nx_options

      assert_all_close(non_linear_eqn_root_nx_options.machine_eps, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.machine_eps) == {:f, 64}

      assert_all_close(non_linear_eqn_root_nx_options.tolerance, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.tolerance) == {:f, 64}

      assert non_linear_eqn_root_nx_options.type == {:f, 64}
      assert non_linear_eqn_root_nx_options.max_iterations == 1_000
      assert non_linear_eqn_root_nx_options.max_fn_eval_count == 1_000
      assert non_linear_eqn_root_nx_options.output_fn_adapter == %ExternalFnAdapter{}
      assert non_linear_eqn_root_nx_options.output_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)
    end

    test "works and does not blow up if t_start and t_end are tensors, not floats" do
      use_default_opts = []
      t_start = Nx.f32(0.0)
      t_end = Nx.f32(10.0)
      order = 5

      nx_options = AdaptiveStepsizeRefactor.convert_to_nx_options(t_start, t_end, order, use_default_opts)
      assert %NxOptions{} = nx_options

      assert nx_options.dt_max == Nx.f32(1.0)
    end

    test "overrides the defaults if provided, including those in the NonLinearEqnRoot.NxOptions" do
      opts = [
        type: :f64,
        max_number_of_errors: 2,
        max_step: 3.0,
        refine: 3,
        speed: 0.5,
        fixed_output_times?: true,
        fixed_output_dt: 0.5,
        norm_control?: false,
        abs_tol: 1.0e-08,
        rel_tol: 1.0e-04,
        output_fn: &Math.sin/1,
        event_fn: &Kernel.max/2,
        zero_fn: &Kernel.min/2,
        #
        # NonLinearEqnRoot.NxOptions:
        machine_eps: 1.0e-03,
        tolerance: 1.0e-04,
        max_iterations: 3,
        max_fn_eval_count: 4,
        nonlinear_eqn_root_output_fn: &Math.sin/1
      ]

      t_start = 0.0
      t_end = 10.0
      order = 3

      nx_options = AdaptiveStepsizeRefactor.convert_to_nx_options(t_start, t_end, order, opts)
      assert %NxOptions{} = nx_options

      assert nx_options.type == {:f, 64}
      assert nx_options.max_number_of_errors == Nx.s32(2)
      assert nx_options.dt_max == Nx.f64(3.0)
      assert nx_options.refine == 3
      assert nx_options.speed == Nx.f64(0.5)
      assert nx_options.fixed_output_times? == Nx.u8(1)
      assert nx_options.fixed_output_dt == Nx.f64(0.5)
      assert nx_options.order == 3
      assert nx_options.norm_control? == Nx.u8(0)
      assert nx_options.abs_tol == Nx.f64(1.0e-08)
      assert nx_options.rel_tol == Nx.f64(1.0e-04)

      assert nx_options.event_fn_adapter == %ExternalFnAdapter{external_fn: &:erlang.max/2}
      assert nx_options.output_fn_adapter == %ExternalFnAdapter{external_fn: &Math.sin/1}
      assert nx_options.zero_fn_adapter == %ExternalFnAdapter{external_fn: &:erlang.min/2}

      # --------------------------------------
      # Values are passed on to NonLinearEqnRoot.NxOptions:
      non_linear_eqn_root_nx_options = nx_options.non_linear_eqn_root_nx_options
      assert %NonLinearEqnRoot.NxOptions{} = non_linear_eqn_root_nx_options

      assert_all_close(non_linear_eqn_root_nx_options.machine_eps, Nx.f64(1.0e-03), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.machine_eps) == {:f, 64}

      assert_all_close(non_linear_eqn_root_nx_options.tolerance, Nx.f64(1.0e-04), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.tolerance) == {:f, 64}

      assert non_linear_eqn_root_nx_options.type == {:f, 64}
      assert non_linear_eqn_root_nx_options.max_iterations == 3
      assert non_linear_eqn_root_nx_options.max_fn_eval_count == 4
      assert non_linear_eqn_root_nx_options.output_fn_adapter == %ExternalFnAdapter{external_fn: &Math.sin/1}
    end
  end
end
