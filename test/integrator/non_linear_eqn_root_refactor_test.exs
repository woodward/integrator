defmodule Integrator.NonLinearEqnRootRefactorTest do
  @moduledoc false
  use Integrator.TestCase
  # import Nx, only: :sigils
  alias Integrator.NonLinearEqnRootRefactor
  alias Integrator.NonLinearEqnRootRefactor.NxOptions

  describe "convert_to_nx_options" do
    test "uses the defaults from nimble options (and defaults for machine_eps and tolerance in the type specified)" do
      opts = []

      nx_options = NonLinearEqnRootRefactor.convert_to_nx_options(opts)
      assert %NxOptions{} = nx_options

      assert_all_close(nx_options.machine_eps, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.machine_eps) == {:f, 64}

      assert_all_close(nx_options.tolerance, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.tolerance) == {:f, 64}

      assert nx_options.type == :f64
      assert nx_options.max_iterations == 1_000
      assert nx_options.max_fn_eval_count == 1_000
      assert nx_options.nonlinear_eqn_root_output_fn == nil
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

      nx_options = NonLinearEqnRootRefactor.convert_to_nx_options(opts)
      assert %NxOptions{} = nx_options

      assert_all_close(nx_options.machine_eps, Nx.tensor(0.1, type: :f32), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.machine_eps) == {:f, 32}

      assert_all_close(nx_options.tolerance, Nx.tensor(0.2, type: :f32), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.tolerance) == {:f, 32}

      assert nx_options.type == :f32
      assert nx_options.max_iterations == 10
      assert nx_options.max_fn_eval_count == 20
      assert nx_options.nonlinear_eqn_root_output_fn == output_fn
    end
  end
end
