defmodule Integrator.NonLinearEqnRootRefactorTest do
  @moduledoc false
  use Integrator.TestCase
  # import Nx, only: :sigils

  alias Integrator.DataCollector
  alias Integrator.NonLinearEqnRoot.InvalidInitialBracketError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError
  alias Integrator.NonLinearEqnRoot.TensorTypeError
  alias Integrator.NonLinearEqnRootRefactor
  alias Integrator.NonLinearEqnRootRefactor.NxOptions

  defmodule NonLinearEqnRootTestFunctions do
    import Nx.Defn

    defn pow_fn(x) do
      Nx.pow(x, 1 / 3) - 1.0e-8
    end
  end

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

      assert Nx.to_number(result.fn_eval_count) == 8
      assert Nx.to_number(result.iteration_count) == 6
      assert Nx.to_number(result.iter_type) == 4

      {x_low, x_high} = NonLinearEqnRootRefactor.bracket_x(result)
      # Expected values are from Octave:
      assert_all_close(x_low, Nx.tensor(3.141592653589793, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.tensor(3.141592653589795, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)

      {y1, y2} = NonLinearEqnRootRefactor.bracket_fx(result)
      # Expected values are from Octave:
      assert_all_close(y1, Nx.tensor(1.224646799147353e-16, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.tensor(-2.097981369335578e-15, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "sine function - works if initial values are swapped" do
      x0 = Nx.tensor(4.0, type: :f64)
      x1 = Nx.tensor(3.0, type: :f64)

      result = NonLinearEqnRootRefactor.find_zero(&Nx.sin/1, x0, x1)

      # Expected value is from Octave:
      expected_x = Nx.tensor(3.141592653589795, type: :f64)
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, Nx.tensor(0.0, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.fn_eval_count) == 8
      assert Nx.to_number(result.iteration_count) == 6
      assert Nx.to_number(result.iter_type) == 4

      {x_low, x_high} = NonLinearEqnRootRefactor.bracket_x(result)
      # Expected values are from Octave:
      assert_all_close(x_low, Nx.tensor(3.141592653589793, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.tensor(3.141592653589795, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)

      {y1, y2} = NonLinearEqnRootRefactor.bracket_fx(result)
      # Expected values are from Octave:
      assert_all_close(y1, Nx.tensor(1.224646799147353e-16, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.tensor(-2.097981369335578e-15, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "sine function - raises an error if invalid initial bracket - positive sine" do
      # Sine is positive for both of these:
      x0 = 2.5
      x1 = 3.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRootRefactor.find_zero(&Nx.sin/1, x0, x1)
      end
    end

    test "sine function - raises an error if invalid initial bracket - negative sine" do
      # Sine is negative for both of these:
      x0 = 3.5
      x1 = 4.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRootRefactor.find_zero(&Nx.sin/1, x0, x1)
      end
    end

    test "sine function - raises an error if max iterations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_iterations: 2]

      assert_raise MaxIterationsExceededError, fn ->
        NonLinearEqnRootRefactor.find_zero(&Nx.sin/1, x0, x1, opts)
      end
    end

    test "sine function - raises an error if max function evaluations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_fn_eval_count: 2]

      assert_raise MaxFnEvalsExceededError, fn ->
        NonLinearEqnRootRefactor.find_zero(&Nx.sin/1, x0, x1, opts)
      end
    end

    test "sine function - outputs values if a function is given" do
      # Octave:
      #   octave> fun = @sin;
      #   octave> x0 = 3;
      #   octave> x1 = 4;
      #   octave> x = fzero(fun, [x0, x1])

      x0 = Nx.tensor(3.0, type: :f64)
      x1 = Nx.tensor(4.0, type: :f64)

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      opts = [nonlinear_eqn_root_output_fn: output_fn]

      result = NonLinearEqnRootRefactor.find_zero(&Nx.sin/1, x0, x1, opts)
      assert_all_close(result.x, Nx.tensor(3.1415926535897936, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, Nx.tensor(-3.216245299353273e-16, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)

      data = DataCollector.get_data(pid)
      assert length(data) == 6

      # From Octave:
      converging_t_data = [
        Nx.tensor(3.157162792479947, type: :f64),
        Nx.tensor(3.141281736699444, type: :f64),
        Nx.tensor(3.141592614571824, type: :f64),
        Nx.tensor(3.141592692610915, type: :f64),
        Nx.tensor(3.141592653589793, type: :f64),
        Nx.tensor(3.141592653589795, type: :f64)
      ]

      t_data = data |> Enum.map(& &1.x)

      assert_nx_lists_equal_refactor(t_data, converging_t_data)
      expected_t = converging_t_data |> Enum.reverse() |> hd()
      assert_all_close(result.x, expected_t, atol: 1.0e-14, rtol: 1.0e-14)

      converged = data |> List.last()

      assert Nx.to_number(converged.iteration_count) == 6
      assert Nx.to_number(converged.fn_eval_count) == 8
      assert_all_close(converged.x, result.x, atol: 1.0e-14, rtol: 1.0e-14)
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

    test "sine function with single initial value (instead of 2)" do
      x0 = Nx.tensor(3.0, type: :f64)

      result = NonLinearEqnRootRefactor.find_zero_with_single_point(&Nx.sin/1, x0)

      # Expected value is from Octave:
      expected_x = Nx.tensor(3.141592653589795, type: :f64)
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, Nx.tensor(0.0, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.fn_eval_count) == 11
      assert Nx.to_number(result.iteration_count) == 4
      assert Nx.to_number(result.iter_type) == 2

      {x_low, x_high} = NonLinearEqnRootRefactor.bracket_x(result)
      # Expected values are from Octave:
      assert_all_close(x_low, Nx.tensor(3.141592653589793, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.tensor(3.141592653589795, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)

      {y1, y2} = NonLinearEqnRootRefactor.bracket_fx(result)
      # Expected values are from Octave:
      assert_all_close(y1, Nx.tensor(1.224646799147353e-16, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.tensor(-2.097981369335578e-15, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "returns pi/2 for cos between 0 & 3 - test from Octave" do
      x0 = Nx.tensor(0.0, type: :f64)
      x1 = Nx.tensor(3.0, type: :f64)

      result = NonLinearEqnRootRefactor.find_zero(&Nx.cos/1, x0, x1)

      expected_x = Nx.divide(Nx.Constants.pi({:f, 64}), Nx.tensor(2.0, type: :f64))
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
    end

    @tag :skip
    test "equation - test from Octave" do
      # Octave (this code is at the bottom of fzero.m):
      #   fun = @(x) x^(1/3) - 1e-8
      #   fzero(fun, [0.0, 1.0])
      x0 = Nx.tensor(0.0, type: :f64)
      x1 = Nx.tensor(1.0, type: :f64)
      zero_fn = &NonLinearEqnRootTestFunctions.pow_fn/1

      result = NonLinearEqnRootRefactor.find_zero(zero_fn, x0, x1)
      dbg(result)

      # Expected values are from Octave:
      # assert_all_close(result.x, Nx.tensor(3.108624468950438e-16, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
      # assert_all_close(result.fx, Nx.tensor(6.764169935169993e-06, type: :f64), atol: 1.0e-22, rtol: 1.0e-22)

      # I am getting these values, but for b - what is up???
      assert_all_close(result.b, Nx.tensor(3.108624468950438e-16, type: :f64), atol: 1.0e-22, rtol: 1.0e-22)
      # And note the decreased precision on this one:
      assert_all_close(result.fb, Nx.tensor(6.764169935169993e-06, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
      #                            actual:  6.76416 7493850112e-6
      # almost looks like single precision agreement
    end

    @tag :skip
    test "staight line through zero - test from Octave" do
      # Octave (this code is at the bottom of fzero.m):
      #   fun = @(x) x
      #   fzero(fun, 0)
      x0 = 0.0
      zero_fn = & &1

      result = NonLinearEqnRootRefactor.find_zero(zero_fn, x0)

      assert_in_delta(result.x, 0.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    @tag :skip
    test "staight line through zero offset by one - test from Octave" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonLinearEqnRootRefactor.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    @tag :skip
    test "staight line through zero offset by one - test from Octave - works" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonLinearEqnRootRefactor.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    @tag :skip
    test "polynomial" do
      # y = (x - 1) * (x - 3) = x^2 - 4*x + 3
      # Roots are 1 and 3

      zero_fn = &(&1 * &1 - 4 * &1 + 3)

      result = NonLinearEqnRootRefactor.find_zero(zero_fn, [0.5, 1.5])

      assert_in_delta(result.x, 1.0, 1.0e-15)
      assert_in_delta(result.fx, 0.0, 1.0e-14)

      result = NonLinearEqnRootRefactor.find_zero(zero_fn, [3.5, 1.5])

      assert_in_delta(result.x, 3.0, 1.0e-15)
      assert_in_delta(result.fx, 0.0, 1.0e-15)
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
