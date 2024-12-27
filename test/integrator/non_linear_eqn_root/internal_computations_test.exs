defmodule Integrator.NonLinearEqnRoot.InternalComputationsTest do
  # Tests of private functions using Patch in NonLinearEqnRoot

  @moduledoc false
  use Integrator.TestCase, async: false
  alias Integrator.NonLinearEqnRootRefactor
  alias Integrator.NonLinearEqnRoot.InternalComputations
  alias Integrator.NonLinearEqnRoot.InternalComputations.SearchFor2ndPoint
  alias Integrator.NonLinearEqnRoot.BracketingFailureError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError

  describe "converged?" do
    # From Octave for:
    # fun = @sin
    # x = fzero(fun, [3, 4])

    test "returns false (i.e., 0) if not yet converged" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141592614571824, type: :f64),
        b: Nx.tensor(3.157162792479947, type: :f64),
        u: Nx.tensor(3.141592614571824, type: :f64)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      assert InternalComputations.converged?(z, machine_epsilon, tolerance) == Nx.tensor(0, type: :u8)
    end

    test "returns true (i.e., 1) if converged" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141592653589793, type: :f64),
        b: Nx.tensor(3.141592653589795, type: :f64),
        u: Nx.tensor(3.141592653589793, type: :f64)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      assert InternalComputations.converged?(z, machine_epsilon, tolerance) == Nx.tensor(1, type: :u8)
    end
  end

  describe "too_far?/1" do
    test "returns true if too far" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.2, type: :f64),
        b: Nx.tensor(3.4, type: :f64),
        u: Nx.tensor(4.0, type: :f64)
      }

      assert InternalComputations.too_far?(Nx.tensor(3.0, type: :f64), z) == Nx.tensor(1, type: :u8)
    end

    test "returns false if not too far" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141592614571824, type: :f64),
        b: Nx.tensor(3.157162792479947, type: :f64),
        u: Nx.tensor(3.141592614571824, type: :f64)
      }

      assert InternalComputations.too_far?(Nx.tensor(3.141592692610915, type: :f64), z) == Nx.tensor(0, type: :u8)
    end
  end

  describe "interpolate" do
    test "bisect" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.0, type: :f64),
        b: Nx.tensor(4.0, type: :f64)
      }

      c = InternalComputations.interpolate_bisect(z)

      assert_all_close(c, Nx.tensor(3.5, type: :f64), atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "double_secant" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141592614571824, type: :f64),
        b: Nx.tensor(3.157162792479947, type: :f64),
        u: Nx.tensor(3.141592614571824, type: :f64),
        fa: Nx.tensor(3.901796897832363e-08, type: :f64),
        fb: Nx.tensor(-1.556950978832860e-02, type: :f64),
        fu: Nx.tensor(3.901796897832363e-08, type: :f64)
      }

      c = InternalComputations.interpolate_double_secant(z)

      assert_all_close(c, Nx.tensor(3.141592692610915, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
    end

    test "quadratic_interpolation_plus_newton" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.0, type: :f64),
        b: Nx.tensor(3.157162792479947, type: :f64),
        d: Nx.tensor(4.0, type: :f64),
        fa: Nx.tensor(0.141120008059867, type: :f64),
        fb: Nx.tensor(-1.556950978832860e-02, type: :f64),
        fd: Nx.tensor(-0.756802495307928, type: :f64),
        fe: Nx.tensor(0.141120008059867, type: :f64),
        iter_type: 2
      }

      c = InternalComputations.interpolate_quadratic_plus_newton(z)

      assert_all_close(c, Nx.tensor(3.141281736699444, type: :f64), atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "quadratic_interpolation_plus_newton - bug fix" do
      # From Octave for ballode - first bounce

      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.995471442091821, type: :f64),
        b: Nx.tensor(4.294180317944318, type: :f64),
        c: Nx.tensor(3.995471442091821, type: :f64),
        d: Nx.tensor(2.898648469921000, type: :f64),
        e: Nx.tensor(4.294180317944318, type: :f64),
        #
        fa: Nx.tensor(1.607028863214206, type: :f64),
        fb: Nx.tensor(-4.564518118928532, type: :f64),
        fc: Nx.tensor(1.607028863214206, type: :f64),
        fd: Nx.tensor(16.76036011799988, type: :f64),
        fe: Nx.tensor(-4.564518118928532, type: :f64),
        #
        iter_type: 2
      }

      c = InternalComputations.interpolate_quadratic_plus_newton(z)

      assert_all_close(c, Nx.tensor(4.077471967384916, type: :f64), atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "inverse_cubic_interpolation" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141281736699444, type: :f64),
        b: Nx.tensor(3.157162792479947, type: :f64),
        d: Nx.tensor(3.0, type: :f64),
        e: Nx.tensor(4.0, type: :f64),
        fa: Nx.tensor(3.109168853400020e-04, type: :f64),
        fb: Nx.tensor(-1.556950978832860e-02, type: :f64),
        fd: Nx.tensor(0.141120008059867, type: :f64),
        fe: Nx.tensor(-0.756802495307928, type: :f64)
      }

      c = InternalComputations.interpolate_inverse_cubic(z)
      assert_all_close(c, Nx.tensor(3.141592614571824, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
    end

    test "secant" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.0, type: :f64),
        b: Nx.tensor(4.0, type: :f64),
        u: Nx.tensor(3.0, type: :f64),
        #
        fa: Nx.tensor(0.141120008059867, type: :f64),
        fb: Nx.tensor(-0.756802495307928, type: :f64),
        fu: Nx.tensor(0.141120008059867, type: :f64)
      }

      c = InternalComputations.interpolate_secant(z)

      assert_all_close(c, Nx.tensor(3.157162792479947, type: :f64), atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  describe "check_for_non_monotonicity/1" do
    test "monotonic" do
      z = %NonLinearEqnRootRefactor{
        d: Nx.tensor(3.141281736699444, type: :f64),
        fa: Nx.tensor(3.901796897832363e-08, type: :f64),
        fb: Nx.tensor(-1.556950978832860e-02, type: :f64),
        fc: Nx.tensor(-3.902112221087341e-08, type: :f64),
        fd: Nx.tensor(3.109168853400020e-04, type: :f64)
      }

      z = InternalComputations.check_for_non_monotonicity(z)

      assert_all_close(z.e, Nx.tensor(3.141281736699444, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
      assert_all_close(z.fe, Nx.tensor(3.109168853400020e-04, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
    end

    test "non-monotonic" do
      z = %NonLinearEqnRootRefactor{
        d: Nx.tensor(3.141281736699444, type: :f64),
        fa: Nx.tensor(-3.911796897832363e-08, type: :f64),
        fb: Nx.tensor(-1.556950978832860e-02, type: :f64),
        fc: Nx.tensor(-3.902112221087341e-08, type: :f64),
        fd: Nx.tensor(3.109168853400020e-04, type: :f64)
      }

      z = InternalComputations.check_for_non_monotonicity(z)

      assert_all_close(z.fe, Nx.tensor(-3.902112221087341e-08, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
    end
  end

  describe "fn_eval_new_point" do
    test "works" do
      z = %NonLinearEqnRootRefactor{
        c: Nx.tensor(3.141281736699444, type: :f64),
        iteration_count: 1,
        fn_eval_count: 3,
        fc: 7
      }

      zero_fn = &Nx.sin/1
      opts = [max_iterations: 1000, max_fn_eval_count: 1000]
      z = InternalComputations.fn_eval_new_point(z, zero_fn, opts)

      assert_all_close(z.fc, Nx.tensor(3.109168853400020e-04, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fx, Nx.tensor(3.109168853400020e-04, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.x, Nx.tensor(3.141281736699444, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)

      assert z.iteration_count == Nx.tensor(2)
      assert z.fn_eval_count == Nx.tensor(4)
    end

    test "raises an error if max iterations exceeded" do
      max_iterations = 4

      z = %NonLinearEqnRootRefactor{
        c: Nx.tensor(3.141281736699444, type: :f64),
        iteration_count: max_iterations,
        fn_eval_count: 3,
        fc: 7
      }

      opts = [max_iterations: max_iterations, max_fn_eval_count: 1000]
      zero_fn = &Nx.sin/1

      assert_raise MaxIterationsExceededError, fn ->
        InternalComputations.fn_eval_new_point(z, zero_fn, opts)
      end
    end

    test "raises an error if max function evaluations exceeded" do
      max_fn_eval_count = 4

      z = %NonLinearEqnRootRefactor{
        c: Nx.tensor(3.141281736699444, type: :f64),
        iteration_count: 1,
        fn_eval_count: max_fn_eval_count,
        fc: 7
      }

      opts = [max_iterations: 1000, max_fn_eval_count: max_fn_eval_count]
      zero_fn = &Nx.sin/1

      assert_raise MaxFnEvalsExceededError, fn ->
        InternalComputations.fn_eval_new_point(z, zero_fn, opts)
      end
    end
  end

  describe "adjust_if_too_close_to_a_or_b" do
    test "when c is NOT too close" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.0, type: :f64),
        b: Nx.tensor(4.0, type: :f64),
        c: Nx.tensor(3.157162792479947, type: :f64),
        u: Nx.tensor(3, type: :f64)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      z = InternalComputations.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance)

      assert_all_close(z.c, Nx.tensor(3.157162792479947, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "when c IS too close" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.157162792479947, type: :f64),
        b: Nx.tensor(3.157162792479948, type: :f64),
        c: Nx.tensor(3.157162792479947, type: :f64),
        u: Nx.tensor(3.157162792479947, type: :f64)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      z = InternalComputations.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance)

      assert_all_close(z.c, Nx.tensor(3.157162792479947, type: :f64), atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  describe "found?/1" do
    test "returns true if signs are different" do
      fa = Nx.tensor(-1.0, type: :f64)
      fb = Nx.tensor(2.0, type: :f64)
      x = %SearchFor2ndPoint{fa: fa, fb: fb}
      assert InternalComputations.found?(x) == Nx.tensor(1, type: :u8)
    end

    test "returns false if signs are the same" do
      fa = Nx.tensor(3.0, type: :f64)
      fb = Nx.tensor(4.0, type: :f64)
      x = %SearchFor2ndPoint{fa: fa, fb: fb}
      assert InternalComputations.found?(x) == Nx.tensor(0, type: :u8)
    end

    test "returns false if signs are the same - 2nd case" do
      fa = Nx.tensor(-5.0, type: :f64)
      fb = Nx.tensor(-6.0, type: :f64)
      x = %SearchFor2ndPoint{fa: fa, fb: fb}
      assert InternalComputations.found?(x) == Nx.tensor(0, type: :u8)
    end
  end

  describe "find_2nd_starting_point" do
    test "finds a value in the vicinity" do
      x0 = Nx.tensor(3.0, type: :f64)
      zero_fn = &Nx.sin/1

      result = InternalComputations.find_2nd_starting_point(zero_fn, x0)

      assert_all_close(result.b, Nx.tensor(3.3, type: :f64), atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(result.fb, Nx.tensor(-0.1577456941432482, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
      assert_all_close(result.fa, Nx.tensor(0.1411200080598672, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)
      assert result.fn_eval_count == Nx.tensor(5, type: :s32)
    end

    test "works if x0 is very close to zero" do
      x0 = Nx.tensor(-0.0005, type: :f64)
      zero_fn = &Nx.sin/1

      result = InternalComputations.find_2nd_starting_point(zero_fn, x0)

      assert_all_close(result.b, Nx.tensor(0.0, type: :f64), atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(result.fb, Nx.tensor(0.0, type: :f64), atol: 1.0e-12, rtol: 1.0e-12)

      # I think there is some single vs double floating point issue here:
      assert_all_close(result.fa, Nx.tensor(-0.09983341664682815, type: :f64), atol: 1.0e-08, rtol: 1.0e-08)
      # If you set the tolerances to 1.0e-09, note these differences which are at the single/double point boundary:
      # -0.09983341 81294999
      # -0.09983341 664682815

      assert result.fn_eval_count == Nx.tensor(8, type: :s32)
    end
  end

  describe "bracket" do
    test "first case - move c down to b" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.Constants.infinity(:f64),
        b: Nx.tensor(3.157162792479947, type: :f64),
        c: Nx.tensor(3.141592692610915, type: :f64),
        #
        fa: Nx.tensor(3.901796897832363e-08, type: :f64),
        fb: Nx.tensor(-1.556950978832860e-02, type: :f64),
        fc: Nx.tensor(-3.902112221087341e-08, type: :f64)
      }

      continue = Nx.tensor(0, type: :s32)
      {^continue, z} = InternalComputations.bracket(z)

      assert_all_close(z.d, Nx.tensor(3.157162792479947, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fd, Nx.tensor(-1.556950978832860e-02, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)

      assert_all_close(z.b, Nx.tensor(3.141592692610915, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fb, Nx.tensor(-3.902112221087341e-08, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "second case - move a up to c" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141281736699444, type: :f64),
        b: Nx.Constants.infinity(:f64),
        c: Nx.tensor(3.141592614571824, type: :f64),
        #
        fa: Nx.tensor(3.109168853400020e-04, type: :f64),
        fb: Nx.tensor(-1.556950978832860e-02, type: :f64),
        fc: Nx.tensor(3.901796897832363e-08, type: :f64)
      }

      continue = Nx.tensor(0, type: :s32)
      {^continue, z} = InternalComputations.bracket(z)

      assert_all_close(z.d, Nx.tensor(3.141281736699444, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fd, Nx.tensor(3.109168853400020e-04, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.a, Nx.tensor(3.141592614571824, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fa, Nx.tensor(3.901796897832363e-08, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "third case - c is already at the root" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.Constants.infinity(:f64),
        b: Nx.Constants.infinity(:f64),
        c: Nx.tensor(1.0, type: :f64),
        #
        fa: Nx.Constants.infinity(:f64),
        fb: Nx.Constants.infinity(:f64),
        fc: Nx.tensor(0.0, type: :f64)
      }

      halt = Nx.tensor(1, type: :s32)
      {^halt, z} = InternalComputations.bracket(z)

      assert_all_close(z.a, Nx.tensor(1.0, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fa, Nx.tensor(0.0, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.b, Nx.tensor(1.0, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fb, Nx.tensor(0.0, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "fourth case - bracket didn't work (note that this is an artificial, non-real-life case)" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.Constants.infinity(:f64),
        b: Nx.Constants.infinity(:f64),
        c: 1.0,
        #
        fa: Nx.Constants.infinity(:f64),
        fb: Nx.Constants.infinity(:f64),
        fc: 0.1
      }

      assert_raise BracketingFailureError, fn ->
        InternalComputations.bracket(z)
      end
    end

    test "bug fix - first iteration of first bounce of ballode.m" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(2.898648469921000, type: :f64),
        b: Nx.tensor(4.294180317944318, type: :f64),
        c: Nx.tensor(3.995471442091821, type: :f64),
        d: Nx.tensor(4.294180317944318, type: :f64),
        #
        fa: Nx.tensor(16.76036011799988, type: :f64),
        fb: Nx.tensor(-4.564518118928532, type: :f64),
        fc: Nx.tensor(1.607028863214206, type: :f64),
        fd: Nx.tensor(-4.564518118928532, type: :f64)
      }

      continue = Nx.tensor(0, type: :s32)
      {^continue, z} = InternalComputations.bracket(z)

      assert_all_close(z.a, Nx.tensor(3.995471442091821, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fa, Nx.tensor(1.607028863214206, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.b, Nx.tensor(4.294180317944318, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fb, Nx.tensor(-4.564518118928532, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.c, Nx.tensor(3.995471442091821, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fc, Nx.tensor(1.607028863214206, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.d, Nx.tensor(2.898648469921000, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fd, Nx.tensor(16.76036011799988, type: :f64), atol: 1.0e-16, rtol: 1.0e-16)
    end
  end
end
