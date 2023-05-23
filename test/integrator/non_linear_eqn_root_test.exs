defmodule Integrator.NonLinearEqnRootTest do
  @moduledoc false
  use Integrator.TestCase
  use Patch
  import Nx, only: :sigils

  alias Integrator.NonLinearEqnRoot.{
    BracketingFailureError,
    InvalidInitialBracketError,
    MaxFnEvalsExceededError,
    MaxIterationsExceededError
  }

  alias Integrator.RungeKutta.DormandPrince45
  alias Integrator.{DummyOutput, NonLinearEqnRoot}

  describe "find_zero" do
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

    test "sine function - raises an error if invalid initial bracket - positive sine" do
      # Sine is positive for both of these:
      x0 = 2.5
      x1 = 3.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])
      end
    end

    test "sine function - raises an error if invalid initial bracket - negative sine" do
      # Sine is negative for both of these:
      x0 = 3.5
      x1 = 4.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])
      end
    end

    test "sine function - raises an error if max iterations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_iterations: 2]

      assert_raise MaxIterationsExceededError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      end
    end

    test "sine function - raises an error if max function evaluations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_fn_eval_count: 2]

      assert_raise MaxFnEvalsExceededError, fn ->
        NonLinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      end
    end

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

    test "returns pi/2 for cos between 0 & 3 - test from Octave" do
      x0 = 0.0
      x1 = 3.0

      result = NonLinearEqnRoot.find_zero(&Math.cos/1, [x0, x1])

      expected_x = Math.pi() / 2.0
      assert_in_delta(result.c, expected_x, 1.0e-14)
    end

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

    test "staight line through zero offset by one - test from Octave" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonLinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    test "staight line through zero offset by one - test from Octave - works" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonLinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

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

    test "ballode - first bounce" do
      # Values obtained from Octave right before and after the call to fzero in ode_event_handler.m:
      t0 = 2.898648469921000
      t1 = 4.294180317944318
      t = Nx.tensor([t0, t1], type: :f64)

      x = ~M[
           1.676036011799988e+01  -4.564518118928532e+00
          -8.435741489925014e+00  -2.212590891903376e+01
      ]f64

      k_vals = ~M[
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
    test "returns a & b" do
      z = %NonLinearEqnRoot{
        a: 3.14,
        b: 3.15
      }

      assert NonLinearEqnRoot.bracket_x(z) == [3.14, 3.15]
    end
  end

  describe "bracket_fx/1" do
    test "returns fa & fb" do
      z = %NonLinearEqnRoot{
        fa: 3.14,
        fb: 3.15
      }

      assert NonLinearEqnRoot.bracket_fx(z) == [3.14, 3.15]
    end
  end

  # ===========================================================================
  # Tests of private functions below here:

  describe "merge_default_opts/1" do
    setup do
      expose(NonLinearEqnRoot, merge_default_opts: 1)

      # assert Example.private_function(:argument) == {:ok, :argument}
    end

    test "returns defaults if no opts are provided" do
      opts = []

      assert private(NonLinearEqnRoot.merge_default_opts(opts)) == [
               machine_eps: 2.220446049250313e-16,
               tolerance: 2.220446049250313e-16,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f64
             ]
    end

    test "use the Nx type for tolerance and machine_eps no opts are provided for those" do
      opts = [type: :f64]

      assert private(NonLinearEqnRoot.merge_default_opts(opts)) == [
               machine_eps: 2.220446049250313e-16,
               tolerance: 2.220446049250313e-16,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f64
             ]

      opts = [type: :f32]

      assert private(NonLinearEqnRoot.merge_default_opts(opts)) == [
               machine_eps: 1.1920928955078125e-07,
               tolerance: 1.1920928955078125e-07,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f32
             ]
    end

    test "use the value for :machine_eps if one is provided" do
      opts = [machine_eps: 1.0e-05]

      assert private(NonLinearEqnRoot.merge_default_opts(opts)) == [
               tolerance: 2.220446049250313e-16,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f64,
               machine_eps: 1.0e-05
             ]
    end

    test "use the value for :tolerance if one is provided" do
      opts = [tolerance: 1.0e-05]

      assert private(NonLinearEqnRoot.merge_default_opts(opts)) == [
               machine_eps: 2.220446049250313e-16,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f64,
               tolerance: 1.0e-05
             ]
    end
  end

  describe "converged?" do
    # From Octave for:
    # fun = @sin
    # x = fzero(fun, [3, 4])

    setup do
      expose(NonLinearEqnRoot, converged?: 3)
    end

    test "returns :continue if not yet converged" do
      z = %NonLinearEqnRoot{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      assert private(NonLinearEqnRoot.converged?(z, machine_epsilon, tolerance)) == :continue
    end

    test "returns :halt if converged" do
      z = %NonLinearEqnRoot{
        a: 3.141592653589793,
        b: 3.141592653589795,
        u: 3.141592653589793
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      assert private(NonLinearEqnRoot.converged?(z, machine_epsilon, tolerance)) == :halt
    end
  end

  describe "interpolate" do
    setup do
      expose(NonLinearEqnRoot, interpolate: 2)
    end

    test "secant" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRoot{
        a: 3,
        b: 4,
        u: 3,
        #
        fa: 0.141120008059867,
        fb: -0.756802495307928,
        fu: 0.141120008059867
      }

      c = private(NonLinearEqnRoot.interpolate(z, :secant))

      assert_in_delta(c, 3.157162792479947, 1.0e-15)
    end

    test "bisect" do
      z = %NonLinearEqnRoot{a: 3, b: 4}
      assert private(NonLinearEqnRoot.interpolate(z, :bisect)) == 3.5
    end

    test "double_secant" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRoot{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824,
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fu: 3.901796897832363e-08
      }

      c = private(NonLinearEqnRoot.interpolate(z, :double_secant))

      assert_in_delta(c, 3.141592692610915, 1.0e-12)
    end

    test "quadratic_interpolation_plus_newton" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRoot{
        a: 3,
        b: 3.157162792479947,
        d: 4,
        fa: 0.141120008059867,
        fb: -1.556950978832860e-02,
        fd: -0.756802495307928,
        fe: 0.141120008059867,
        iter_type: 2
      }

      c = private(NonLinearEqnRoot.interpolate(z, :quadratic_interpolation_plus_newton))

      assert_in_delta(c, 3.141281736699444, 1.0e-15)
    end

    test "quadratic_interpolation_plus_newton - bug fix" do
      # From Octave for ballode - first bounce

      z = %NonLinearEqnRoot{
        a: 3.995471442091821,
        b: 4.294180317944318,
        c: 3.995471442091821,
        d: 2.898648469921000,
        e: 4.294180317944318,
        #
        fa: 1.607028863214206,
        fb: -4.564518118928532,
        fc: 1.607028863214206,
        fd: 16.76036011799988,
        fe: -4.564518118928532,
        #
        iter_type: 2
      }

      c = private(NonLinearEqnRoot.interpolate(z, :quadratic_interpolation_plus_newton))

      assert_in_delta(c, 4.077471967384916, 1.0e-15)
    end

    test "inverse_cubic_interpolation" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRoot{
        a: 3.141281736699444,
        b: 3.157162792479947,
        d: 3.0,
        e: 4.0,
        fa: 3.109168853400020e-04,
        fb: -1.556950978832860e-02,
        fd: 0.141120008059867,
        fe: -0.756802495307928
      }

      c = private(NonLinearEqnRoot.interpolate(z, :inverse_cubic_interpolation))
      assert_in_delta(c, 3.141592614571824, 1.0e-12)
    end
  end

  describe "too_far?/1" do
    setup do
      expose(NonLinearEqnRoot, too_far?: 2)
    end

    test "returns true if too far" do
      z = %NonLinearEqnRoot{
        a: 3.2,
        b: 3.4,
        u: 4.0
      }

      assert private(NonLinearEqnRoot.too_far?(3.0, z)) == true
    end

    test "returns false if not too far" do
      z = %NonLinearEqnRoot{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824
      }

      assert private(NonLinearEqnRoot.too_far?(3.141592692610915, z)) == false
    end
  end

  describe "check_for_non_monotonicity/1" do
    setup do
      expose(NonLinearEqnRoot, check_for_non_monotonicity: 1)
    end

    test "monotonic" do
      z = %NonLinearEqnRoot{
        d: 3.141281736699444,
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08,
        fd: 3.109168853400020e-04
      }

      z = private(NonLinearEqnRoot.check_for_non_monotonicity(z))
      assert_in_delta(z.e, 3.141281736699444, 1.0e-12)
      assert_in_delta(z.fe, 3.109168853400020e-04, 1.0e-12)
    end

    test "non-monotonic" do
      z = %NonLinearEqnRoot{
        d: 3.141281736699444,
        fa: -3.911796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08,
        fd: 3.109168853400020e-04
      }

      z = private(NonLinearEqnRoot.check_for_non_monotonicity(z))
      assert_in_delta(z.fe, -3.902112221087341e-08, 1.0e-12)
    end
  end

  describe "fn_eval_new_point" do
    setup do
      expose(NonLinearEqnRoot, fn_eval_new_point: 3)
    end

    test "works" do
      z = %NonLinearEqnRoot{
        c: 3.141281736699444,
        iteration_count: 1,
        fn_eval_count: 3,
        fc: 7
      }

      zero_fn = &Math.sin/1
      opts = [max_iterations: 1000]
      z = private(NonLinearEqnRoot.fn_eval_new_point(z, zero_fn, opts))

      assert_in_delta(z.fc, 3.109168853400020e-04, 1.0e-16)
      assert_in_delta(z.fx, 3.109168853400020e-04, 1.0e-16)
      assert_in_delta(z.x, 3.141281736699444, 1.0e-16)

      assert z.iteration_count == 2
      assert z.fn_eval_count == 4
    end

    test "raises an error if max iterations exceeded" do
      max_iterations = 4

      z = %NonLinearEqnRoot{
        c: 3.141281736699444,
        iteration_count: max_iterations,
        fn_eval_count: 3,
        fc: 7
      }

      opts = [max_iterations: max_iterations]
      zero_fn = &Math.sin/1

      assert_raise MaxIterationsExceededError, fn ->
        private(NonLinearEqnRoot.fn_eval_new_point(z, zero_fn, opts))
      end
    end
  end

  describe "adjust_if_too_close_to_a_or_b" do
    setup do
      expose(NonLinearEqnRoot, adjust_if_too_close_to_a_or_b: 3)
    end

    test "when c is NOT too close" do
      z = %NonLinearEqnRoot{
        a: 3.0,
        b: 4.0,
        c: 3.157162792479947,
        u: 3
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      z = private(NonLinearEqnRoot.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance))

      assert_in_delta(z.c, 3.157162792479947, 1.0e-16)
    end

    test "when c IS too close" do
      z = %NonLinearEqnRoot{
        a: 3.157162792479947,
        b: 3.157162792479948,
        c: 3.157162792479947,
        u: 3.157162792479947
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      z = private(NonLinearEqnRoot.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance))

      assert_in_delta(z.c, 3.157162792479947, 1.0e-15)
    end
  end

  describe "find_2nd_starting_point" do
    setup do
      expose(NonLinearEqnRoot, find_2nd_starting_point: 2)
    end

    test "finds a value in the vicinity" do
      x0 = 3.0

      result = private(NonLinearEqnRoot.find_2nd_starting_point(&Math.sin/1, x0))

      assert_in_delta(result.b, 3.3, 1.0e-15)
      assert_in_delta(result.fb, -0.1577456941432482, 1.0e-12)
      assert_in_delta(result.fa, 0.1411200080598672, 1.0e-12)
      assert result.fn_eval_count == 5
    end

    test "works if x0 is very close to zero" do
      x0 = -0.0005

      result = private(NonLinearEqnRoot.find_2nd_starting_point(&Math.sin/1, x0))

      assert_in_delta(result.b, 0.0, 1.0e-15)
      assert_in_delta(result.fb, 0.0, 1.0e-12)
      assert_in_delta(result.fa, -0.09983341664682815, 1.0e-12)
      assert result.fn_eval_count == 8
    end
  end

  describe "bracket" do
    setup do
      expose(NonLinearEqnRoot, bracket: 1)
    end

    test "first case - move b down to c" do
      z = %NonLinearEqnRoot{
        a: nil,
        b: 3.157162792479947,
        c: 3.141592692610915,
        #
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08
      }

      {:continue, z} = private(NonLinearEqnRoot.bracket(z))

      assert z.d == 3.157162792479947
      assert z.fd == -1.556950978832860e-02

      assert z.b == 3.141592692610915
      assert z.fb == -3.902112221087341e-08
    end

    test "second case - move a up to c" do
      z = %NonLinearEqnRoot{
        a: 3.141281736699444,
        b: nil,
        c: 3.141592614571824,
        #
        fa: 3.109168853400020e-04,
        fb: -1.556950978832860e-02,
        fc: 3.901796897832363e-08
      }

      {:continue, z} = private(NonLinearEqnRoot.bracket(z))

      assert z.d == 3.141281736699444
      assert z.fd == 3.109168853400020e-04

      assert z.a == 3.141592614571824
      assert z.fa == 3.901796897832363e-08
    end

    test "third case - c is already at the root" do
      z = %NonLinearEqnRoot{
        a: nil,
        b: nil,
        c: 1.0,
        #
        fa: nil,
        fb: nil,
        fc: 0.0
      }

      {:halt, z} = private(NonLinearEqnRoot.bracket(z))

      assert z.a == 1.0
      assert z.fa == 0.0

      assert z.b == 1.0
      assert z.fb == 0.0
    end

    test "fourth case - bracket didn't work (note that this is an artificial, non-real-life case)" do
      z = %NonLinearEqnRoot{
        a: nil,
        b: nil,
        c: 1.0,
        #
        fa: nil,
        fb: nil,
        fc: 0.1
      }

      assert_raise BracketingFailureError, fn ->
        private(NonLinearEqnRoot.bracket(z))
      end
    end

    test "bug fix - first iteration of first bounce of ballode.m" do
      z = %NonLinearEqnRoot{
        a: 2.898648469921000,
        b: 4.294180317944318,
        c: 3.995471442091821,
        d: 4.294180317944318,
        #
        fa: 16.76036011799988,
        fb: -4.564518118928532,
        fc: 1.607028863214206,
        fd: -4.564518118928532
      }

      {:continue, z} = private(NonLinearEqnRoot.bracket(z))

      assert z.a == 3.995471442091821
      assert z.fa == 1.607028863214206

      assert z.b == 4.294180317944318
      assert z.fb == -4.564518118928532

      assert z.c == 3.995471442091821
      assert z.fc == 1.607028863214206

      assert z.d == 2.898648469921000
      assert z.fd == 16.76036011799988
    end
  end

  describe "compute_iteration_two_or_three" do
    setup do
      expose(NonLinearEqnRoot, compute_iteration_two_or_three: 1)
    end

    test "bug fix" do
      z = %NonLinearEqnRoot{
        a: 3.995471442091821,
        b: 4.077471967384916,
        c: 4.077471967384916,
        d: 4.294180317944318,
        e: 2.898648469921000,
        #
        fa: 1.607028863214206,
        fb: -9.382095100818333e-11,
        fc: -9.382095100818333e-11,
        fd: -4.564518118928532,
        fe: 16.76036011799988,
        #
        iter_type: 2
      }

      z = private(NonLinearEqnRoot.compute_iteration_two_or_three(z))

      assert z.a == 3.995471442091821
      assert z.b == 4.077471967384916
      assert z.c == 4.077471967380238
      assert z.d == 4.294180317944318
      assert z.e == 2.898648469921000

      assert z.fa == 1.607028863214206
      assert z.fb == -9.382095100818333e-11
      assert z.fc == -9.382095100818333e-11
      assert z.fd == -4.564518118928532
      assert z.fe == 16.76036011799988
    end
  end
end
