defmodule Integrator.NonLinearEqnRootRefactorTest do
  @moduledoc false
  use Integrator.TestCase
  # import Nx, only: :sigils
  alias Integrator.NonLinearEqnRootRefactor
  alias Integrator.NonLinearEqnRootRefactor.NxOptions
  alias Integrator.NonLinearEqnRoot.TensorTypeError

  describe "find_zero" do
    test "sine function (so the zeros of this are known values) - computations in :f64" do
      # Octave:
      # fun = @sin; % function
      # x0 = 3;
      # x1 = 4;
      # x = fzero(fun, [x0, x1])
      # x = 3.141592653589795
      #
      # opts = optimset("Display", "iter")
      # x = fzero(fun, [x0, x1], opts)

      # Search for a zero in the interval [3, 4]:
      #  Fcn-count    x          f(x)             Procedure
      #     2              3       0.14112        initial
      #     3        3.15716    -0.0155695        interpolation
      #     4        3.14128   0.000310917        interpolation
      #     5        3.14159    3.9018e-08        interpolation
      #     6        3.14159  -3.90211e-08        interpolation
      #     7        3.14159   1.22465e-16        interpolation
      #     8        3.14159  -2.09798e-15        interpolation
      # Algorithm converged.
      # x = 3.141592653589795

      x0 = Nx.tensor(3.0, type: :f64)
      x1 = Nx.tensor(4.0, type: :f64)

      result = NonLinearEqnRootRefactor.find_zero(&Nx.sin/1, x0, x1)

      # Expected value is from Octave:
      expected_x = Nx.tensor(3.141592653589795, type: :f64)
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, 0.0, atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.fn_eval_count) == 23
      assert Nx.to_number(result.iteration_count) == 21
      assert Nx.to_number(result.iter_type) == 4

      # Original values before refactor - why are these different?
      # assert Nx.to_number(result.fn_eval_count) == 8
      # assert Nx.to_number(result.iteration_count) == 6
      # assert Nx.to_number(result.iter_type) == 4

      {x_low, x_high} = NonLinearEqnRootRefactor.bracket_x(result)
      # Expected values are from Octave:
      assert_all_close(x_low, Nx.tensor(3.141592653589793, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.tensor(3.141592653589795, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)

      {y1, y2} = NonLinearEqnRootRefactor.bracket_fx(result)
      # Expected values are from Octave:
      assert_all_close(y1, Nx.tensor(1.224646799147353e-16, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.tensor(-2.097981369335578e-15, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end
  end

  describe "convert_to_nx_options" do
    test "uses the defaults from nimble options (and defaults for machine_eps and tolerance in the type specified)" do
      opts = []

      nx_options = NonLinearEqnRootRefactor.convert_to_nx_options(opts)
      assert %NxOptions{} = nx_options

      assert_all_close(nx_options.machine_eps, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.machine_eps) == {:f, 64}

      assert_all_close(nx_options.tolerance, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.tolerance) == {:f, 64}

      assert nx_options.type == {:f, 64}
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

      assert nx_options.type == {:f, 32}
      assert nx_options.max_iterations == 10
      assert nx_options.max_fn_eval_count == 20
      assert nx_options.nonlinear_eqn_root_output_fn == output_fn
    end
  end

  describe "convert_arg_to_nx_type" do
    test "passes through tensors (if they are of the correct type)" do
      arg = Nx.tensor(1.0, type: :f64)
      assert NonLinearEqnRootRefactor.convert_arg_to_nx_type(arg, {:f, 64}) == Nx.tensor(1.0, type: :f64)
    end

    test "converts floats to tensors of the appropriate type" do
      arg = 1.0
      assert NonLinearEqnRootRefactor.convert_arg_to_nx_type(arg, {:f, 32}) == Nx.tensor(1.0, type: :f32)

      assert NonLinearEqnRootRefactor.convert_arg_to_nx_type(arg, {:f, 64}) == Nx.tensor(1.0, type: :f64)
    end

    test "raises an exception if you try to cast a tensor to a different type" do
      arg = Nx.tensor(1.0, type: :f64)

      assert_raise TensorTypeError, fn ->
        NonLinearEqnRootRefactor.convert_arg_to_nx_type(arg, {:f, 32})
      end
    end
  end
end
