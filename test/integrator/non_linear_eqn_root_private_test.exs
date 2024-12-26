defmodule Integrator.NonLinearEqnRootPrivateTest do
  # Tests of private functions using Patch in NonLinearEqnRoot

  @moduledoc false
  use Integrator.TestCase, async: false
  use Patch

  alias Integrator.NonLinearEqnRoot
  alias Integrator.NonLinearEqnRoot.BracketingFailureError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError

  describe "merge_default_opts/1" do
    setup do
      expose(NonLinearEqnRoot, merge_default_opts: 1)

      # assert Example.private_function(:argument) == {:ok, :argument}
    end

    @tag transferred_to_refactor?: false
    test "returns defaults if no opts are provided" do
      opts = []

      {:ok, default_opts} = NimbleOptions.validate(opts, NonLinearEqnRoot.options_schema())

      assert private(NonLinearEqnRoot.merge_default_opts(default_opts)) == [
               machine_eps: 2.220446049250313e-16,
               tolerance: 2.220446049250313e-16,
               nonlinear_eqn_root_output_fn: nil,
               type: :f64,
               max_fn_eval_count: 1000,
               max_iterations: 1000
             ]
    end

    @tag transferred_to_refactor?: false
    test "use the Nx type for tolerance and machine_eps no opts are provided for those" do
      opts = [type: :f64]

      {:ok, merged_opts} = NimbleOptions.validate(opts, NonLinearEqnRoot.options_schema())

      assert private(NonLinearEqnRoot.merge_default_opts(merged_opts)) == [
               machine_eps: 2.220446049250313e-16,
               tolerance: 2.220446049250313e-16,
               nonlinear_eqn_root_output_fn: nil,
               max_fn_eval_count: 1000,
               max_iterations: 1000,
               type: :f64
             ]

      opts = [type: :f32]
      {:ok, merged_opts} = NimbleOptions.validate(opts, NonLinearEqnRoot.options_schema())

      assert private(NonLinearEqnRoot.merge_default_opts(merged_opts)) == [
               machine_eps: 1.1920928955078125e-7,
               tolerance: 1.1920928955078125e-7,
               nonlinear_eqn_root_output_fn: nil,
               max_fn_eval_count: 1000,
               max_iterations: 1000,
               type: :f32
             ]
    end

    @tag transferred_to_refactor?: false
    test "use the value for :machine_eps if one is provided" do
      opts = [machine_eps: 1.0e-05]
      {:ok, merged_opts} = NimbleOptions.validate(opts, NonLinearEqnRoot.options_schema())

      assert private(NonLinearEqnRoot.merge_default_opts(merged_opts)) == [
               tolerance: 2.220446049250313e-16,
               nonlinear_eqn_root_output_fn: nil,
               type: :f64,
               max_fn_eval_count: 1000,
               max_iterations: 1000,
               machine_eps: 1.0e-05
             ]
    end

    @tag transferred_to_refactor?: false
    test "use the value for :tolerance if one is provided" do
      opts = [tolerance: 1.0e-05]
      {:ok, merged_opts} = NimbleOptions.validate(opts, NonLinearEqnRoot.options_schema())

      assert private(NonLinearEqnRoot.merge_default_opts(merged_opts)) == [
               machine_eps: 2.220446049250313e-16,
               nonlinear_eqn_root_output_fn: nil,
               type: :f64,
               max_fn_eval_count: 1000,
               max_iterations: 1000,
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

    @tag transferred_to_refactor?: true
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

    @tag transferred_to_refactor?: true
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
    test "bisect" do
      z = %NonLinearEqnRoot{a: 3, b: 4}
      assert private(NonLinearEqnRoot.interpolate(z, :bisect)) == 3.5
    end

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
    test "returns true if too far" do
      z = %NonLinearEqnRoot{
        a: 3.2,
        b: 3.4,
        u: 4.0
      }

      assert private(NonLinearEqnRoot.too_far?(3.0, z)) == true
    end

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
    test "finds a value in the vicinity" do
      x0 = 3.0

      result = private(NonLinearEqnRoot.find_2nd_starting_point(&Math.sin/1, x0))

      assert_in_delta(result.b, 3.3, 1.0e-15)
      assert_in_delta(result.fb, -0.1577456941432482, 1.0e-12)
      assert_in_delta(result.fa, 0.1411200080598672, 1.0e-12)
      assert result.fn_eval_count == 5
    end

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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
