defmodule Integrator.AdaptiveStepsize.NxOptionsTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.AdaptiveStepsize.NxOptions
  alias Integrator.ExternalFnAdapter
  alias Integrator.NonLinearEqnRoot

  describe "convert_to_nx_options" do
    test "uses the defaults from nimble options (and defaults for machine_eps and tolerance in the type specified)" do
      use_default_opts = []
      t_start = 0.0
      t_end = 10.0
      order = 5

      nx_options = NxOptions.convert_opts_to_nx_options(t_start, t_end, order, use_default_opts)
      assert %NxOptions{} = nx_options

      assert nx_options.type == {:f, 32}
      assert nx_options.max_number_of_errors == Nx.s32(5_000)
      assert nx_options.max_step == Nx.f32(1.0)
      assert nx_options.refine == 4
      assert nx_options.speed == Nx.Constants.infinity(:f32)
      assert nx_options.fixed_output_times? == Nx.u8(0)
      assert nx_options.fixed_output_step == Nx.f32(0.0)
      assert nx_options.order == 5
      assert nx_options.norm_control? == Nx.u8(1)
      assert nx_options.abs_tol == Nx.f32(1.0e-06)
      assert nx_options.rel_tol == Nx.f32(1.0e-03)
      assert nx_options.nx_while_loop_integration? == Nx.u8(1)

      assert nx_options.event_fn_adapter == %ExternalFnAdapter{external_fn: &Integrator.ExternalFnAdapter.no_op_double_arity_fn/2}
      assert nx_options.output_fn_adapter == %ExternalFnAdapter{external_fn: &Integrator.ExternalFnAdapter.non_defn_no_op_fn/1}

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

      nx_options = NxOptions.convert_opts_to_nx_options(t_start, t_end, order, use_default_opts)
      assert %NxOptions{} = nx_options

      assert nx_options.max_step == Nx.f32(1.0)
    end

    test "sets :refine to 1 if using fixed sizes, regardless of the value" do
      opts = [refine: 4, fixed_output_times?: true]
      t_start = Nx.f32(0.0)
      t_end = Nx.f32(10.0)
      order = 5

      nx_options = NxOptions.convert_opts_to_nx_options(t_start, t_end, order, opts)
      assert %NxOptions{} = nx_options

      assert nx_options.refine == 1
    end

    test "overrides the defaults if provided, including those in the NonLinearEqnRoot.NxOptions" do
      opts = [
        type: :f64,
        max_number_of_errors: 2,
        max_step: 3.0,
        refine: 3,
        speed: 0.5,
        fixed_output_times?: true,
        fixed_output_step: 0.5,
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

      nx_options = NxOptions.convert_opts_to_nx_options(t_start, t_end, order, opts)
      assert %NxOptions{} = nx_options

      assert nx_options.type == {:f, 64}
      assert nx_options.max_number_of_errors == Nx.s32(2)
      assert nx_options.max_step == Nx.f64(3.0)
      assert nx_options.refine == 1
      assert nx_options.speed == Nx.f64(0.5)
      assert nx_options.fixed_output_times? == Nx.u8(1)
      assert nx_options.fixed_output_step == Nx.f64(0.5)
      assert nx_options.order == 3
      assert nx_options.norm_control? == Nx.u8(0)
      assert nx_options.abs_tol == Nx.f64(1.0e-08)
      assert nx_options.rel_tol == Nx.f64(1.0e-04)
      assert nx_options.nx_while_loop_integration? == Nx.u8(0)

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
