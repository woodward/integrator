defmodule Integrator.NonLinearEqnRoot.InternalComputationsTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.NonLinearEqnRoot
  alias Integrator.NonLinearEqnRoot.InternalComputations
  alias Integrator.NonLinearEqnRoot.InternalComputations.SearchFor2ndPoint
  alias Integrator.NonLinearEqnRoot.BracketingFailureError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError

  defmodule TestFunctions do
    @moduledoc false
    import Nx.Defn

    defn sin(x, _args), do: Nx.sin(x)
  end

  describe "converged?" do
    # From Octave for:
    # fun = @sin
    # x = fzero(fun, [3, 4])

    test "returns false (i.e., 0) if not yet converged" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.141592614571824),
        b: Nx.f64(3.157162792479947),
        u: Nx.f64(3.141592614571824)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      assert InternalComputations.converged?(z, machine_epsilon, tolerance) == Nx.u8(0)
    end

    test "returns true (i.e., 1) if converged" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.141592653589793),
        b: Nx.f64(3.141592653589795),
        u: Nx.f64(3.141592653589793)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      assert InternalComputations.converged?(z, machine_epsilon, tolerance) == Nx.u8(1)
    end
  end

  describe "too_far?/1" do
    test "returns true if too far" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.2),
        b: Nx.f64(3.4),
        u: Nx.f64(4.0)
      }

      assert InternalComputations.too_far?(Nx.f64(3.0), z) == Nx.u8(1)
    end

    test "returns false if not too far" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.141592614571824),
        b: Nx.f64(3.157162792479947),
        u: Nx.f64(3.141592614571824)
      }

      assert InternalComputations.too_far?(Nx.f64(3.141592692610915), z) == Nx.u8(0)
    end
  end

  describe "check_for_non_monotonicity/1" do
    test "monotonic" do
      z = %NonLinearEqnRoot{
        d: Nx.f64(3.141281736699444),
        fa: Nx.f64(3.901796897832363e-08),
        fb: Nx.f64(-1.556950978832860e-02),
        fc: Nx.f64(-3.902112221087341e-08),
        fd: Nx.f64(3.109168853400020e-04)
      }

      z = InternalComputations.check_for_non_monotonicity(z)

      assert_all_close(z.e, Nx.f64(3.141281736699444), atol: 1.0e-12, rtol: 1.0e-12)
      assert_all_close(z.fe, Nx.f64(3.109168853400020e-04), atol: 1.0e-12, rtol: 1.0e-12)
    end

    test "non-monotonic" do
      z = %NonLinearEqnRoot{
        d: Nx.f64(3.141281736699444),
        fa: Nx.f64(-3.911796897832363e-08),
        fb: Nx.f64(-1.556950978832860e-02),
        fc: Nx.f64(-3.902112221087341e-08),
        fd: Nx.f64(3.109168853400020e-04)
      }

      z = InternalComputations.check_for_non_monotonicity(z)

      assert_all_close(z.fe, Nx.f64(-3.902112221087341e-08), atol: 1.0e-12, rtol: 1.0e-12)
    end
  end

  describe "fn_eval_new_point" do
    test "works" do
      z = %NonLinearEqnRoot{
        c: Nx.f64(3.141281736699444),
        iteration_count: 1,
        fn_eval_count: 3,
        fc: 7
      }

      zero_fn = &TestFunctions.sin/2
      zero_fn_args = []
      opts = %{max_iterations: 1000, max_fn_eval_count: 1000}
      z = InternalComputations.fn_eval_new_point(z, zero_fn, zero_fn_args, opts)

      assert_all_close(z.fc, Nx.f64(3.109168853400020e-04), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fx, Nx.f64(3.109168853400020e-04), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.x, Nx.f64(3.141281736699444), atol: 1.0e-16, rtol: 1.0e-16)

      assert z.iteration_count == Nx.s32(2)
      assert z.fn_eval_count == Nx.s32(4)
    end

    test "raises an error if max iterations exceeded" do
      max_iterations = 4

      z = %NonLinearEqnRoot{
        c: Nx.f64(3.141281736699444),
        iteration_count: max_iterations,
        fn_eval_count: 3,
        fc: 7
      }

      opts = %{max_iterations: max_iterations, max_fn_eval_count: 1000}
      zero_fn = &TestFunctions.sin/2
      zero_fn_args = []

      assert_raise MaxIterationsExceededError, fn ->
        InternalComputations.fn_eval_new_point(z, zero_fn, zero_fn_args, opts)
      end
    end

    test "raises an error if max function evaluations exceeded" do
      max_fn_eval_count = 4

      z = %NonLinearEqnRoot{
        c: Nx.f64(3.141281736699444),
        iteration_count: 1,
        fn_eval_count: max_fn_eval_count,
        fc: 7
      }

      opts = %{max_iterations: 1000, max_fn_eval_count: max_fn_eval_count}
      zero_fn = &TestFunctions.sin/2
      zero_fn_args = []

      assert_raise MaxFnEvalsExceededError, fn ->
        InternalComputations.fn_eval_new_point(z, zero_fn, zero_fn_args, opts)
      end
    end
  end

  describe "adjust_if_too_close_to_a_or_b" do
    test "when c is NOT too close" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.0),
        b: Nx.f64(4.0),
        c: Nx.f64(3.157162792479947),
        u: Nx.f64(3)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      z = InternalComputations.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance)

      assert_all_close(z.c, Nx.f64(3.157162792479947), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "when c IS too close" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.157162792479947),
        b: Nx.f64(3.157162792479948),
        c: Nx.f64(3.157162792479947),
        u: Nx.f64(3.157162792479947)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      z = InternalComputations.adjust_if_too_close_to_a_or_b(z, machine_epsilon, tolerance)

      assert_all_close(z.c, Nx.f64(3.157162792479947), atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  describe "found?/1" do
    test "returns true if signs are different" do
      fa = Nx.f64(-1.0)
      fb = Nx.f64(2.0)
      x = %SearchFor2ndPoint{fa: fa, fb: fb}
      assert InternalComputations.found?(x) == Nx.u8(1)
    end

    test "returns false if signs are the same" do
      fa = Nx.f64(3.0)
      fb = Nx.f64(4.0)
      x = %SearchFor2ndPoint{fa: fa, fb: fb}
      assert InternalComputations.found?(x) == Nx.u8(0)
    end

    test "returns false if signs are the same - 2nd case" do
      fa = Nx.f64(-5.0)
      fb = Nx.f64(-6.0)
      x = %SearchFor2ndPoint{fa: fa, fb: fb}
      assert InternalComputations.found?(x) == Nx.u8(0)
    end
  end

  describe "find_2nd_starting_point" do
    test "finds a value in the vicinity" do
      x0 = Nx.f64(3.0)
      zero_fn = &TestFunctions.sin/2
      zero_fn_args = []

      result = InternalComputations.find_2nd_starting_point(zero_fn, x0, zero_fn_args)

      assert_all_close(result.b, Nx.f64(3.3), atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(result.fb, Nx.f64(-0.1577456941432482), atol: 1.0e-12, rtol: 1.0e-12)
      assert_all_close(result.fa, Nx.f64(0.1411200080598672), atol: 1.0e-12, rtol: 1.0e-12)
      assert result.fn_eval_count == Nx.s32(5)
    end

    test "works if x0 is very close to zero" do
      x0 = Nx.f64(-0.0005)
      zero_fn = &TestFunctions.sin/2
      zero_fn_args = []

      result = InternalComputations.find_2nd_starting_point(zero_fn, x0, zero_fn_args)

      assert_all_close(result.b, Nx.f64(0.0), atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(result.fb, Nx.f64(0.0), atol: 1.0e-12, rtol: 1.0e-12)

      # I think there is some single vs double floating point issue here:
      assert_all_close(result.fa, Nx.f64(-0.09983341664682815), atol: 1.0e-08, rtol: 1.0e-08)
      # If you set the tolerances to 1.0e-09, note these differences which are at the single/double point boundary:
      # -0.09983341 81294999
      # -0.09983341 664682815

      assert result.fn_eval_count == Nx.s32(8)
    end
  end

  describe "bracket" do
    test "first case - move c down to b" do
      z = %NonLinearEqnRoot{
        a: Nx.Constants.nan(:f64),
        b: Nx.f64(3.157162792479947),
        c: Nx.f64(3.141592692610915),
        #
        fa: Nx.f64(3.901796897832363e-08),
        fb: Nx.f64(-1.556950978832860e-02),
        fc: Nx.f64(-3.902112221087341e-08)
      }

      continue = Nx.s32(1)
      {^continue, z} = InternalComputations.bracket(z)

      assert_all_close(z.d, Nx.f64(3.157162792479947), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fd, Nx.f64(-1.556950978832860e-02), atol: 1.0e-16, rtol: 1.0e-16)

      assert_all_close(z.b, Nx.f64(3.141592692610915), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fb, Nx.f64(-3.902112221087341e-08), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "second case - move a up to c" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.141281736699444),
        b: Nx.Constants.nan(:f64),
        c: Nx.f64(3.141592614571824),
        #
        fa: Nx.f64(3.109168853400020e-04),
        fb: Nx.f64(-1.556950978832860e-02),
        fc: Nx.f64(3.901796897832363e-08)
      }

      continue = Nx.s32(1)
      {^continue, z} = InternalComputations.bracket(z)

      assert_all_close(z.d, Nx.f64(3.141281736699444), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fd, Nx.f64(3.109168853400020e-04), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.a, Nx.f64(3.141592614571824), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fa, Nx.f64(3.901796897832363e-08), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "third case - c is already at the root" do
      z = %NonLinearEqnRoot{
        a: Nx.Constants.nan(:f64),
        b: Nx.Constants.nan(:f64),
        c: Nx.f64(1.0),
        #
        fa: Nx.Constants.nan(:f64),
        fb: Nx.Constants.nan(:f64),
        fc: Nx.f64(0.0)
      }

      halt = Nx.s32(0)
      {^halt, z} = InternalComputations.bracket(z)

      assert_all_close(z.a, Nx.f64(1.0), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fa, Nx.f64(0.0), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.b, Nx.f64(1.0), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fb, Nx.f64(0.0), atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "fourth case - bracket didn't work (note that this is an artificial, non-real-life case)" do
      z = %NonLinearEqnRoot{
        a: Nx.Constants.nan(:f64),
        b: Nx.Constants.nan(:f64),
        c: 1.0,
        #
        fa: Nx.Constants.nan(:f64),
        fb: Nx.Constants.nan(:f64),
        fc: 0.1
      }

      assert_raise BracketingFailureError, fn ->
        InternalComputations.bracket(z)
      end
    end

    test "bug fix - first iteration of first bounce of ballode.m" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(2.898648469921000),
        b: Nx.f64(4.294180317944318),
        c: Nx.f64(3.995471442091821),
        d: Nx.f64(4.294180317944318),
        #
        fa: Nx.f64(16.76036011799988),
        fb: Nx.f64(-4.564518118928532),
        fc: Nx.f64(1.607028863214206),
        fd: Nx.f64(-4.564518118928532)
      }

      continue = Nx.s32(1)
      {^continue, z} = InternalComputations.bracket(z)

      assert_all_close(z.a, Nx.f64(3.995471442091821), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fa, Nx.f64(1.607028863214206), atol: 1.0e-16, rtol: 1.0e-16)

      assert_all_close(z.b, Nx.f64(4.294180317944318), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fb, Nx.f64(-4.564518118928532), atol: 1.0e-16, rtol: 1.0e-16)

      assert_all_close(z.c, Nx.f64(3.995471442091821), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fc, Nx.f64(1.607028863214206), atol: 1.0e-16, rtol: 1.0e-16)

      assert_all_close(z.d, Nx.f64(2.898648469921000), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fd, Nx.f64(16.76036011799988), atol: 1.0e-16, rtol: 1.0e-16)
    end
  end

  describe "compute_iteration_types_2_or_3" do
    test "bug fix" do
      z = %NonLinearEqnRoot{
        a: Nx.f64(3.995471442091821),
        b: Nx.f64(4.077471967384916),
        c: Nx.f64(4.077471967384916),
        d: Nx.f64(4.294180317944318),
        e: Nx.f64(2.898648469921000),
        #
        fa: Nx.f64(1.607028863214206),
        fb: Nx.f64(-9.382095100818333e-11),
        fc: Nx.f64(-9.382095100818333e-11),
        fd: Nx.f64(-4.564518118928532),
        fe: Nx.f64(16.76036011799988),
        #
        iteration_type: 2
      }

      z = InternalComputations.compute_iteration_types_2_or_3(z)

      assert_all_close(z.a, Nx.f64(3.995471442091821), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.b, Nx.f64(4.077471967384916), atol: 1.0e-16, rtol: 1.0e-16)

      # Precision issues?
      # 4.07747196738 4916
      # 4.07747196738 0238
      assert_all_close(z.c, Nx.f64(4.077471967380238), atol: 1.0e-12, rtol: 1.0e-12)

      assert_all_close(z.d, Nx.f64(4.294180317944318), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.e, Nx.f64(2.898648469921000), atol: 1.0e-16, rtol: 1.0e-16)

      assert_all_close(z.fa, Nx.f64(1.607028863214206), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fb, Nx.f64(-9.382095100818333e-11), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fc, Nx.f64(-9.382095100818333e-11), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fd, Nx.f64(-4.564518118928532), atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(z.fe, Nx.f64(16.76036011799988), atol: 1.0e-16, rtol: 1.0e-16)
    end
  end

  describe "number_of_unique_values" do
    test "returns 4 if all values are unique" do
      assert InternalComputations.number_of_unique_values(1, 2, 3, 4) == Nx.u8(4)
    end

    test "returns 3 if all but one values are unique" do
      assert InternalComputations.number_of_unique_values(1, 2, 3, 1) == Nx.u8(3)
      assert InternalComputations.number_of_unique_values(1, 2, 3, 3) == Nx.u8(3)
      assert InternalComputations.number_of_unique_values(1, 2, 2, 4) == Nx.u8(3)
      assert InternalComputations.number_of_unique_values(1, 1, 3, 4) == Nx.u8(3)
    end

    test "returns 2 if two values are unique" do
      assert InternalComputations.number_of_unique_values(1, 2, 2, 1) == Nx.u8(2)
      assert InternalComputations.number_of_unique_values(1, 2, 1, 2) == Nx.u8(2)
      assert InternalComputations.number_of_unique_values(1, 1, 2, 2) == Nx.u8(2)
    end

    test "returns 1 if only value is unique" do
      assert InternalComputations.number_of_unique_values(1, 1, 1, 1) == Nx.u8(1)
    end
  end
end
