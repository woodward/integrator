defmodule Integrator.NonLinearEqnRootTest do
  @moduledoc false
  use Integrator.TestCase
  import Nx, only: :sigils

  alias Integrator.DummyOutput
  alias Integrator.NonLinearEqnRoot
  alias Integrator.NonLinearEqnRoot.InvalidInitialBracketError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError
  alias Integrator.RungeKutta.DormandPrince45

  describe "find_zero" do
    @tag transferred_to_refactor?: true
    test "sine function" do
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

      x0 = 3.0
      x1 = 4.0

      result = NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])

      # Expected value is from Octave:
      expected_x = 3.141592653589795
      assert_in_delta(result.c, expected_x, 1.0e-14)
      assert_in_delta(result.fx, 0.0, 1.0e-14)

      assert result.fn_eval_count == 8
      assert result.iteration_count == 6
      assert result.iter_type == 4

      [x_low, x_high] = NonLinearEqnRoot.bracket_x(result)
      # Expected values are from Octave:
      assert_in_delta(x_low, 3.141592653589793, 1.0e-14)
      assert_in_delta(x_high, 3.141592653589795, 1.0e-14)

      [y1, y2] = NonLinearEqnRoot.bracket_fx(result)
      # Expected values are from Octave:
      assert_in_delta(y1, 1.224646799147353e-16, 1.0e-14)
      assert_in_delta(y2, -2.097981369335578e-15, 1.0e-14)
    end

    @tag transferred_to_refactor?: true
    test "sine function - works if initial values are swapped" do
      x0 = 4.0
      x1 = 3.0

      result = NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])

      # Expected value is from Octave:
      expected_x = 3.141592653589795
      assert_in_delta(result.c, expected_x, 1.0e-14)
      assert_in_delta(result.fx, 0.0, 1.0e-14)

      assert result.fn_eval_count == 8
      assert result.iteration_count == 6
      assert result.iter_type == 4

      [x_low, x_high] = NonLinearEqnRoot.bracket_x(result)
      # Expected values are from Octave:
      assert_in_delta(x_low, 3.141592653589793, 1.0e-14)
      assert_in_delta(x_high, 3.141592653589795, 1.0e-14)

      [y1, y2] = NonLinearEqnRoot.bracket_fx(result)
      # Expected values are from Octave:
      assert_in_delta(y1, 1.224646799147353e-16, 1.0e-14)
      assert_in_delta(y2, -2.097981369335578e-15, 1.0e-14)
    end

    @tag transferred_to_refactor?: true
    test "sine function - raises an error if invalid initial bracket - positive sine" do
      # Sine is positive for both of these:
      x0 = 2.5
      x1 = 3.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])
      end
    end

    @tag transferred_to_refactor?: true
    test "sine function - raises an error if invalid initial bracket - negative sine" do
      # Sine is negative for both of these:
      x0 = 3.5
      x1 = 4.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])
      end
    end

    @tag transferred_to_refactor?: true
    test "sine function - raises an error if max iterations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_iterations: 2]

      assert_raise MaxIterationsExceededError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      end
    end

    @tag transferred_to_refactor?: true
    test "sine function - raises an error if max function evaluations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_fn_eval_count: 2]

      assert_raise MaxFnEvalsExceededError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      end
    end

    @tag transferred_to_refactor?: true
    test "sine function - outputs values if a function is given" do
      # Octave:
      #   octave> fun = @sin;
      #   octave> x0 = 3;
      #   octave> x1 = 4;
      #   octave> x = fzero(fun, [x0, x1])

      x0 = 3.0
      x1 = 4.0

      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, step -> DummyOutput.add_data(dummy_output_name, %{t: t, x: step}) end

      opts = [nonlinear_eqn_root_output_fn: output_fn]

      result = NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      assert_in_delta(result.x, 3.1415926535897936, 1.0e-14)
      assert_in_delta(result.fx, -3.216245299353273e-16, 1.0e-14)

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 6
      assert length(t_data) == 6

      # From Octave:
      converging_t_data = [
        3.157162792479947,
        3.141281736699444,
        3.141592614571824,
        3.141592692610915,
        3.141592653589793,
        3.141592653589795
      ]

      assert_lists_equal(t_data, converging_t_data, 1.0e-15)
      expected_t = converging_t_data |> Enum.reverse() |> hd()
      assert_in_delta(result.x, expected_t, 1.0e-14)

      converged = x_data |> Enum.reverse() |> hd()

      assert converged.iteration_count == 6
      assert converged.fn_eval_count == 8
      assert_in_delta(converged.x, result.x, 1.0e-14)
    end

    @tag transferred_to_refactor?: true
    test "sine function with single initial value (instead of 2)" do
      x0 = 3.0

      result = NonLinearEqnRoot.find_zero(&Math.sin/1, x0)

      # Expected value is from Octave:
      expected_x = 3.141592653589795
      assert_in_delta(result.c, expected_x, 1.0e-14)
      assert_in_delta(result.fx, 0.0, 1.0e-14)

      assert result.fn_eval_count == 11
      assert result.iteration_count == 4
      assert result.iter_type == 2

      [x_low, x_high] = NonLinearEqnRoot.bracket_x(result)
      # Expected values are from Octave:
      assert_in_delta(x_low, 3.141592653589793, 1.0e-14)
      assert_in_delta(x_high, 3.141592653589795, 1.0e-14)

      [y1, y2] = NonLinearEqnRoot.bracket_fx(result)
      # Expected values are from Octave:
      assert_in_delta(y1, 1.224646799147353e-16, 1.0e-14)
      assert_in_delta(y2, -2.097981369335578e-15, 1.0e-14)
    end

    @tag transferred_to_refactor?: true
    test "returns pi/2 for cos between 0 & 3 - test from Octave" do
      x0 = 0.0
      x1 = 3.0

      result = NonLinearEqnRoot.find_zero(&Math.cos/1, [x0, x1])

      expected_x = Math.pi() / 2.0
      assert_in_delta(result.c, expected_x, 1.0e-14)
    end

    @tag transferred_to_refactor?: true
    test "equation - test from Octave" do
      # Octave (this code is at the bottom of fzero.m):
      #   fun = @(x) x^(1/3) - 1e-8
      #   fzero(fun, [0.0, 1.0])
      x0 = 0.0
      x1 = 1.0
      zero_fn = &(Math.pow(&1, 1 / 3) - 1.0e-8)

      result = NonLinearEqnRoot.find_zero(zero_fn, [x0, x1])

      # Expected values are from Octave:
      assert_in_delta(result.x, 3.108624468950438e-16, 1.0e-24)
      assert_in_delta(result.fx, 6.764169935169993e-06, 1.0e-22)
    end

    @tag transferred_to_refactor?: true
    test "staight line through zero - test from Octave" do
      # Octave (this code is at the bottom of fzero.m):
      #   fun = @(x) x
      #   fzero(fun, 0)
      x0 = 0.0
      zero_fn = & &1

      result = NonLinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, 0.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    @tag transferred_to_refactor?: true
    test "staight line through zero offset by one - test from Octave" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonLinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    @tag transferred_to_refactor?: true
    test "staight line through zero offset by one - test from Octave - works" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonLinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    @tag transferred_to_refactor?: true
    test "polynomial" do
      # y = (x - 1) * (x - 3) = x^2 - 4*x + 3
      # Roots are 1 and 3

      zero_fn = &(&1 * &1 - 4 * &1 + 3)

      result = NonLinearEqnRoot.find_zero(zero_fn, [0.5, 1.5])

      assert_in_delta(result.x, 1.0, 1.0e-15)
      assert_in_delta(result.fx, 0.0, 1.0e-14)

      result = NonLinearEqnRoot.find_zero(zero_fn, [3.5, 1.5])

      assert_in_delta(result.x, 3.0, 1.0e-15)
      assert_in_delta(result.fx, 0.0, 1.0e-15)
    end

    @tag transferred_to_refactor?: true
    test "ballode - first bounce" do
      # Values obtained from Octave right before and after the call to fzero in ode_event_handler.m:
      t0 = 2.898648469921000
      t1 = 4.294180317944318
      t = Nx.tensor([t0, t1], type: :f64)

      x = ~MAT[
           1.676036011799988e+01  -4.564518118928532e+00
          -8.435741489925014e+00  -2.212590891903376e+01
      ]f64

      k_vals = ~MAT[
          -8.435741489925014e+00  -1.117377497574676e+01  -1.254279171865764e+01  -1.938787543321202e+01  -2.060477920468836e+01   -2.212590891903378e+01  -2.212590891903376e+01
          -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00   -9.810000000000000e+00  -9.810000000000000e+00
      ]f64

      zero_fn = fn t_out ->
        x_out = DormandPrince45.interpolate(t, x, k_vals, Nx.tensor(t_out, type: :f64))
        Nx.to_number(x_out[0][0])
      end

      result = NonLinearEqnRoot.find_zero(zero_fn, [t0, t1])

      # Expected value is from Octave:
      expected_x = 4.077471967380223
      assert_in_delta(result.c, expected_x, 1.0e-14)
      # This should be close to zero because we found the zero root:
      assert_in_delta(result.fx, 0.0, 1.0e-13)

      assert result.fn_eval_count == 7
      assert result.iteration_count == 5
      assert result.iter_type == 3

      [x__low, x_high] = NonLinearEqnRoot.bracket_x(result)
      # Expected values are from Octave; note that these are the same except in the last digit:
      assert_in_delta(x__low, 4.077471967380224, 1.0e-14)
      assert_in_delta(x_high, 4.077471967380227, 1.0e-14)
      # Octave:
      # 4.077471967380223
      # 4.077471967380223

      [y_1, y2] = NonLinearEqnRoot.bracket_fx(result)
      assert_in_delta(y_1, 0.0, 1.0e-14)
      assert_in_delta(y2, 0.0, 1.0e-14)
      # In Octave:
      # [0, 0]

      x_out = DormandPrince45.interpolate(t, x, k_vals, Nx.tensor(result.c, type: :f64))
      assert_in_delta(Nx.to_number(x_out[0][0]), 0.0, 1.0e-15)
      assert_in_delta(Nx.to_number(x_out[1][0]), -20.0, 1.0e-14)
    end
  end

  describe "bracket_x/1" do
    @tag transferred_to_refactor?: true
    test "returns a & b" do
      z = %NonLinearEqnRoot{
        a: 3.14,
        b: 3.15
      }

      assert NonLinearEqnRoot.bracket_x(z) == [3.14, 3.15]
    end
  end

  describe "bracket_fx/1" do
    @tag transferred_to_refactor?: true
    test "returns fa & fb" do
      z = %NonLinearEqnRoot{
        fa: 3.14,
        fb: 3.15
      }

      assert NonLinearEqnRoot.bracket_fx(z) == [3.14, 3.15]
    end
  end

  describe "option_keys" do
    @tag transferred_to_refactor?: false
    test "returns the option keys" do
      assert NonLinearEqnRoot.option_keys() == [
               :nonlinear_eqn_root_output_fn,
               :type,
               :max_fn_eval_count,
               :max_iterations
             ]
    end
  end
end
