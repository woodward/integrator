defmodule Integrator.NonLinearEqnRoot.NxOptionsTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.ExternalFnAdapter
  alias Integrator.NonLinearEqnRoot.NxOptions

  describe "convert_to_nx_options" do
    test "uses the defaults from nimble options (and defaults for machine_eps and tolerance in the type specified)" do
      opts = []

      nx_options = NxOptions.convert_opts_to_nx_options(opts)
      assert %NxOptions{} = nx_options

      assert_all_close(nx_options.machine_eps, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.machine_eps) == {:f, 64}

      assert_all_close(nx_options.tolerance, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.tolerance) == {:f, 64}

      assert nx_options.type == {:f, 64}
      assert nx_options.max_iterations == 1_000
      assert nx_options.max_fn_eval_count == 1_000
      assert nx_options.output_fn_adapter == %ExternalFnAdapter{}
      assert nx_options.output_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)
    end

    test "allows overrides" do
      output_fn = fn x -> x end

      opts = [
        type: :f32,
        max_iterations: 10,
        max_fn_eval_count: 20,
        machine_eps: 0.1,
        tolerance: 0.2,
        nonlinear_eqn_root_output_fn: output_fn
      ]

      nx_options = NxOptions.convert_opts_to_nx_options(opts)
      assert %NxOptions{} = nx_options

      assert_all_close(nx_options.machine_eps, Nx.f32(0.1), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.machine_eps) == {:f, 32}

      assert_all_close(nx_options.tolerance, Nx.f32(0.2), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.tolerance) == {:f, 32}

      assert nx_options.type == {:f, 32}
      assert nx_options.max_iterations == 10
      assert nx_options.max_fn_eval_count == 20
      assert nx_options.output_fn_adapter.external_fn == output_fn
    end
  end
end
