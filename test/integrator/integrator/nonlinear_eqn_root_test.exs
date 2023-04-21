defmodule Integrator.NonlinearEqnRootTest do
  @moduledoc false
  use Integrator.DemoCase
  use Patch

  alias Integrator.NonlinearEqnRoot.{
    BracketingFailureError,
    InvalidInitialBracketError,
    MaxFnEvalsExceededError,
    MaxIterationsExceededError
  }

  alias Integrator.{DummyOutput, NonlinearEqnRoot}

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

      result = NonlinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])

      # Expected value is from Octave:
      expected_x = 3.141592653589795
      assert_in_delta(result.c, expected_x, 1.0e-14)
      assert_in_delta(result.fx, 0.0, 1.0e-15)

      assert result.fn_eval_count == 8
      assert result.iteration_count == 6
      assert result.iter_type == 4

      [x_low, x_high] = result.bracket_x
      # Expected values are from Octave:
      assert_in_delta(x_low, 3.141592653589793, 1.0e-14)
      assert_in_delta(x_high, 3.141592653589795, 1.0e-14)

      [y1, y2] = result.bracket_fx
      # Expected values are from Octave:
      assert_in_delta(y1, 1.224646799147353e-16, 1.0e-14)
      assert_in_delta(y2, -2.097981369335578e-15, 1.0e-14)
    end

    test "sine function - works if initial values are swapped" do
      x0 = 4.0
      x1 = 3.0

      result = NonlinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])

      # Expected value is from Octave:
      expected_x = 3.141592653589795
      assert_in_delta(result.c, expected_x, 1.0e-14)
      assert_in_delta(result.fx, 0.0, 1.0e-15)

      assert result.fn_eval_count == 8
      assert result.iteration_count == 6
      assert result.iter_type == 4

      [x_low, x_high] = result.bracket_x
      # Expected values are from Octave:
      assert_in_delta(x_low, 3.141592653589793, 1.0e-14)
      assert_in_delta(x_high, 3.141592653589795, 1.0e-14)

      [y1, y2] = result.bracket_fx
      # Expected values are from Octave:
      assert_in_delta(y1, 1.224646799147353e-16, 1.0e-14)
      assert_in_delta(y2, -2.097981369335578e-15, 1.0e-14)
    end

    test "sine function - raises an error if invalid initial bracket - positive sine" do
      # Sine is positive for both of these:
      x0 = 2.5
      x1 = 3.0

      assert_raise InvalidInitialBracketError, fn ->
        NonlinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])
      end
    end

    test "sine function - raises an error if invalid initial bracket - negative sine" do
      # Sine is negative for both of these:
      x0 = 3.5
      x1 = 4.0

      assert_raise InvalidInitialBracketError, fn ->
        NonlinearEqnRoot.find_zero(&Math.sin/1, [x0, x1])
      end
    end

    test "sine function - raises an error if max iterations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_iterations: 2]

      assert_raise MaxIterationsExceededError, fn ->
        NonlinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      end
    end

    test "sine function - raises an error if max function evaluations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_fn_eval_count: 2]

      assert_raise MaxFnEvalsExceededError, fn ->
        NonlinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      end
    end

    test "sine function - outputs values if a function is given" do
      x0 = 3.0
      x1 = 4.0

      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, step -> DummyOutput.add_data(dummy_output_name, %{t: t, x: step}) end

      opts = [nonlinear_eqn_root_output_fn: output_fn]

      result = NonlinearEqnRoot.find_zero(&Math.sin/1, [x0, x1], opts)
      assert result.x == 3.1415926535897936
      assert result.fx == -3.216245299353273e-16

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 7
      assert length(t_data) == 7

      # IO.inspect(t_data)

      converging_t_data = [
        3.0,
        3.157162792479947,
        3.157162792479945,
        3.141596389566289,
        3.141596389566289,
        # Above value is repeated and should be this (below); figure out why it's repeated:
        # 3.1415888925885564,
        3.1415926535897936,
        # Why does the last point show up twice?
        3.1415926535897936
      ]

      assert_lists_equal(t_data, converging_t_data, 1.0e-15)
      expected_t = converging_t_data |> Enum.reverse() |> hd()
      assert_in_delta(result.x, expected_t, 1.0e-14)

      converged = x_data |> Enum.reverse() |> hd()

      assert converged.iteration_count == 6
      assert converged.fn_eval_count == 8
      assert_in_delta(converged.x, result.x, 1.0e-14)

      first = x_data |> hd()
      expected_bracket_x = [3.0, 4.0]
      assert_lists_equal(first.bracket_x, expected_bracket_x, 1.0e-15)

      expected_bracket_fx = [0.1411200080598672, -0.7568024953079282]
      assert_lists_equal(first.bracket_fx, expected_bracket_fx, 1.0e-15)

      assert_in_delta(first.x, 3.0, 1.0e-15)
      assert_in_delta(first.fx, 0.1411200080598672, 1.0e-15)
    end

    test "sine function with single initial value (instead of 2)" do
      x0 = 3.0

      result = NonlinearEqnRoot.find_zero(&Math.sin/1, x0)

      # Expected value is from Octave:
      expected_x = 3.141592653589795
      assert_in_delta(result.c, expected_x, 1.0e-14)
      assert_in_delta(result.fx, 0.0, 1.0e-15)

      assert result.fn_eval_count == 13
      assert result.iteration_count == 6
      assert result.iter_type == 4

      [x_low, x_high] = result.bracket_x
      # Expected values are from Octave:
      assert_in_delta(x_low, 3.141592653589793, 1.0e-14)
      assert_in_delta(x_high, 3.141592653589795, 1.0e-14)

      [y1, y2] = result.bracket_fx
      # Expected values are from Octave:
      assert_in_delta(y1, 1.224646799147353e-16, 1.0e-14)
      assert_in_delta(y2, -2.097981369335578e-15, 1.0e-14)
    end

    test "returns pi/2 for cos between 0 & 3 - test from Octave" do
      x0 = 0.0
      x1 = 3.0

      result = NonlinearEqnRoot.find_zero(&Math.cos/1, [x0, x1])

      expected_x = Math.pi() / 2.0
      assert_in_delta(result.c, expected_x, 1.0e-15)
    end

    test "equation - test from Octave" do
      x0 = 0.0
      x1 = 1.0
      zero_fn = &(Math.pow(&1, 1 / 3) - 1.0e-8)

      result = NonlinearEqnRoot.find_zero(zero_fn, [x0, x1])

      assert_in_delta(result.x, 1.0e-24, 1.0e-22)
      assert_in_delta(result.fx, -1.0e-08, 1.0e-22)
    end

    test "staight line through zero - test from Octave" do
      x0 = 0.0
      zero_fn = & &1

      result = NonlinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, 0.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    test "staight line through zero offset by one - test from Octave" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonlinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    test "staight line through zero offset by one - test from Octave - works" do
      x0 = 0.0
      zero_fn = &(&1 + 1)

      result = NonlinearEqnRoot.find_zero(zero_fn, x0)

      assert_in_delta(result.x, -1.0, 1.0e-22)
      assert_in_delta(result.fx, 0.0, 1.0e-22)
    end

    test "polynomial" do
      # y = (x - 1) * (x - 3) = x^2 - 4*x + 3
      # Roots are 1 and 3

      zero_fn = &(&1 * &1 - 4 * &1 + 3)

      result = NonlinearEqnRoot.find_zero(zero_fn, [0.5, 1.5])

      assert_in_delta(result.x, 1.0, 1.0e-15)
      assert_in_delta(result.fx, 0.0, 1.0e-15)

      result = NonlinearEqnRoot.find_zero(zero_fn, [3.5, 1.5])

      assert_in_delta(result.x, 3.0, 1.0e-15)
      assert_in_delta(result.fx, 0.0, 1.0e-15)
    end
  end

  # ===========================================================================
  # Tests of private functions below here:

  describe "merge_default_opts/1" do
    setup do
      expose(NonlinearEqnRoot, merge_default_opts: 1)

      # assert Example.private_function(:argument) == {:ok, :argument}
    end

    test "returns defaults if no opts are provided" do
      opts = []

      assert private(NonlinearEqnRoot.merge_default_opts(opts)) == [
               machine_eps: 2.220446049250313e-16,
               tolerance: 2.220446049250313e-16,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f64
             ]
    end

    test "use the Nx type for tolerance and machine_eps no opts are provided for those" do
      opts = [type: :f64]

      assert private(NonlinearEqnRoot.merge_default_opts(opts)) == [
               machine_eps: 2.220446049250313e-16,
               tolerance: 2.220446049250313e-16,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f64
             ]

      opts = [type: :f32]

      assert private(NonlinearEqnRoot.merge_default_opts(opts)) == [
               machine_eps: 1.1920929e-7,
               tolerance: 1.1920929e-7,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f32
             ]
    end

    test "use the value for :machine_eps if one is provided" do
      opts = [machine_eps: 1.0e-05]

      assert private(NonlinearEqnRoot.merge_default_opts(opts)) == [
               tolerance: 2.220446049250313e-16,
               max_iterations: 1000,
               max_fn_eval_count: 1000,
               type: :f64,
               machine_eps: 1.0e-05
             ]
    end

    test "use the value for :tolerance if one is provided" do
      opts = [tolerance: 1.0e-05]

      assert private(NonlinearEqnRoot.merge_default_opts(opts)) == [
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
      expose(NonlinearEqnRoot, converged?: 3)
    end

    test "returns :continue if not yet converged" do
      z = %NonlinearEqnRoot{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      assert private(NonlinearEqnRoot.converged?(z, machine_epsilon, tolerance)) == :continue
    end

    test "returns :halt if converged" do
      z = %NonlinearEqnRoot{
        a: 3.141592653589793,
        b: 3.141592653589795,
        u: 3.141592653589793
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      assert private(NonlinearEqnRoot.converged?(z, machine_epsilon, tolerance)) == :halt
    end
  end

  describe "interpolate" do
    setup do
      expose(NonlinearEqnRoot, interpolate: 2)
    end

    test "secant" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonlinearEqnRoot{
        a: 3,
        b: 4,
        u: 3,
        #
        fa: 0.141120008059867,
        fb: -0.756802495307928,
        fu: 0.141120008059867
      }

      c = private(NonlinearEqnRoot.interpolate(z, :secant))

      assert_in_delta(c, 3.157162792479947, 1.0e-15)
    end

    test "bisect" do
      z = %NonlinearEqnRoot{a: 3, b: 4}
      assert private(NonlinearEqnRoot.interpolate(z, :bisect)) == 3.5
    end

    test "double_secant" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonlinearEqnRoot{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824,
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fu: 3.901796897832363e-08
      }

      c = private(NonlinearEqnRoot.interpolate(z, :double_secant))

      assert_in_delta(c, 3.141592692610915, 1.0e-12)
    end

    test "quadratic_interpolation_plus_newton" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonlinearEqnRoot{
        a: 3,
        b: 3.157162792479947,
        d: 4,
        fa: 0.141120008059867,
        fb: -1.556950978832860e-02,
        fd: -0.756802495307928,
        fe: 0.141120008059867,
        iter_type: 2
      }

      c = private(NonlinearEqnRoot.interpolate(z, :quadratic_interpolation_plus_newton))

      assert_in_delta(c, 3.141281736699444, 1.0e-15)
    end

    test "inverse_cubic_interpolation" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonlinearEqnRoot{
        a: 3.141281736699444,
        b: 3.157162792479947,
        d: 3.0,
        e: 4.0,
        fa: 3.109168853400020e-04,
        fb: -1.556950978832860e-02,
        fd: 0.141120008059867,
        fe: -0.756802495307928
      }

      c = private(NonlinearEqnRoot.interpolate(z, :inverse_cubic_interpolation))
      assert_in_delta(c, 3.141592614571824, 1.0e-12)
    end
  end

  describe "too_far?/1" do
    setup do
      expose(NonlinearEqnRoot, too_far?: 2)
    end

    test "returns true if too far" do
      z = %NonlinearEqnRoot{
        a: 3.2,
        b: 3.4,
        u: 4.0
      }

      assert private(NonlinearEqnRoot.too_far?(3.0, z)) == true
    end

    test "returns false if not too far" do
      z = %NonlinearEqnRoot{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824
      }

      assert private(NonlinearEqnRoot.too_far?(3.141592692610915, z)) == false
    end
  end

  describe "check_for_non_monotonicity/1" do
    setup do
      expose(NonlinearEqnRoot, check_for_non_monotonicity: 1)
    end

    test "monotonic" do
      z = %NonlinearEqnRoot{
        d: 3.141281736699444,
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08,
        fd: 3.109168853400020e-04
      }

      z = private(NonlinearEqnRoot.check_for_non_monotonicity(z))
      assert_in_delta(z.e, 3.141281736699444, 1.0e-12)
      assert_in_delta(z.fe, 3.109168853400020e-04, 1.0e-12)
    end

    test "non-monotonic" do
      z = %NonlinearEqnRoot{
        d: 3.141281736699444,
        fa: -3.911796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08,
        fd: 3.109168853400020e-04
      }

      z = private(NonlinearEqnRoot.check_for_non_monotonicity(z))
      assert_in_delta(z.fe, -3.902112221087341e-08, 1.0e-12)
    end
  end

  describe "fn_eval_new_point" do
    setup do
      expose(NonlinearEqnRoot, fn_eval_new_point: 3)
    end

    test "works" do
      z = %NonlinearEqnRoot{
        c: 3.141281736699444,
        iteration_count: 1,
        fn_eval_count: 3,
        fc: 7
      }

      zero_fn = &Math.sin/1
      opts = [max_iterations: 1000]
      z = private(NonlinearEqnRoot.fn_eval_new_point(z, zero_fn, opts))

      assert_in_delta(z.fc, 3.109168853400020e-04, 1.0e-16)
      assert_in_delta(z.fx, 3.109168853400020e-04, 1.0e-16)
      assert_in_delta(z.x, 3.141281736699444, 1.0e-16)

      assert z.iteration_count == 2
      assert z.fn_eval_count == 4
    end

    test "raises an error if max iterations exceeded" do
      max_iterations = 4

      z = %NonlinearEqnRoot{
        c: 3.141281736699444,
        iteration_count: max_iterations,
        fn_eval_count: 3,
        fc: 7
      }

      opts = [max_iterations: max_iterations]
      zero_fn = &Math.sin/1

      assert_raise MaxIterationsExceededError, fn ->
        private(NonlinearEqnRoot.fn_eval_new_point(z, zero_fn, opts))
      end
    end
  end

  describe "adjust_if_too_close_to_a_or_b" do
    setup do
      expose(NonlinearEqnRoot, adjust_if_too_close_to_a_or_b: 3)
    end

    test "when c is NOT too close" do
      z = %NonlinearEqnRoot{
        a: 3.0,
        b: 4.0,
        c: 3.157162792479947,
        u: 3
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      z = private(NonlinearEqnRoot.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance))

      assert_in_delta(z.c, 3.157162792479947, 1.0e-16)
    end

    test "when c IS too close" do
      z = %NonlinearEqnRoot{
        a: 3.157162792479947,
        b: 3.157162792479948,
        c: 3.157162792479947,
        u: 3.157162792479947
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      z = private(NonlinearEqnRoot.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance))

      assert_in_delta(z.c, 3.157162792479947, 1.0e-15)
    end
  end

  describe "find_2nd_starting_point" do
    setup do
      expose(NonlinearEqnRoot, find_2nd_starting_point: 2)
    end

    test "finds a value in the vicinity" do
      x0 = 3.0

      result = private(NonlinearEqnRoot.find_2nd_starting_point(&Math.sin/1, x0))

      assert_in_delta(result.b, 3.3, 1.0e-15)
      assert_in_delta(result.fb, -0.1577456941432482, 1.0e-12)
      assert_in_delta(result.fa, 0.1411200080598672, 1.0e-12)
      assert result.fn_eval_count == 5
    end

    test "works if x0 is very close to zero" do
      x0 = -0.0005

      result = private(NonlinearEqnRoot.find_2nd_starting_point(&Math.sin/1, x0))

      assert_in_delta(result.b, 0.0, 1.0e-15)
      assert_in_delta(result.fb, 0.0, 1.0e-12)
      assert_in_delta(result.fa, -0.09983341664682815, 1.0e-12)
      assert result.fn_eval_count == 8
    end
  end

  describe "set_x_results/1" do
    setup do
      expose(NonlinearEqnRoot, set_x_results: 1)
    end

    test "sets the appropriate values on x, fx, bracket_x, and bracket_fx" do
      z = %NonlinearEqnRoot{
        a: 2,
        b: 4,
        u: 3,
        fa: 1,
        fb: 2,
        fu: 1.5
      }

      z = private(NonlinearEqnRoot.set_x_results(z))

      assert z.x == 3
      assert z.fx == 1.5
      assert z.bracket_x == [2, 4]
      assert z.bracket_fx == [1, 2]
    end
  end

  describe "bracket" do
    setup do
      expose(NonlinearEqnRoot, bracket: 1)
    end

    test "first case - move b down to c" do
      z = %NonlinearEqnRoot{
        a: nil,
        b: 3.157162792479947,
        c: 3.141592692610915,
        #
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08
      }

      {:continue, z} = private(NonlinearEqnRoot.bracket(z))

      assert z.d == 3.157162792479947
      assert z.fd == -1.556950978832860e-02

      assert z.b == 3.141592692610915
      assert z.fb == -3.902112221087341e-08
    end

    test "second case - move a up to c" do
      z = %NonlinearEqnRoot{
        a: 3.141281736699444,
        b: nil,
        c: 3.141592614571824,
        #
        fa: 3.109168853400020e-04,
        fb: -1.556950978832860e-02,
        fc: 3.901796897832363e-08
      }

      {:continue, z} = private(NonlinearEqnRoot.bracket(z))

      assert z.d == 3.141281736699444
      assert z.fd == 3.109168853400020e-04

      assert z.a == 3.141592614571824
      assert z.fa == 3.901796897832363e-08
    end

    test "third case - c is already at the root" do
      z = %NonlinearEqnRoot{
        a: nil,
        b: nil,
        c: 1.0,
        #
        fa: nil,
        fb: nil,
        fc: 0.0
      }

      {:halt, z} = private(NonlinearEqnRoot.bracket(z))

      assert z.a == 1.0
      assert z.fa == 0.0

      assert z.b == 1.0
      assert z.fb == 0.0
    end

    test "fourth case - bracket didn't work (note that this is an artificial, non-real-life case)" do
      z = %NonlinearEqnRoot{
        a: nil,
        b: nil,
        c: 1.0,
        #
        fa: nil,
        fb: nil,
        fc: 0.1
      }

      assert_raise BracketingFailureError, fn ->
        private(NonlinearEqnRoot.bracket(z))
      end
    end
  end
end
