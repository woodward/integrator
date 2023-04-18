defmodule Integrator.EqnZeroSolverTest do
  @moduledoc false
  use Integrator.DemoCase
  import Nx, only: :sigils

  alias Integrator.{Utils, EqnZeroSolver}

  describe "fzero" do
    @tag :skip
    test "works" do
      t_old = 2.155396117711071
      t_new = 2.742956500140625

      y_old = ~V[ 1.283429405203074e-02 -2.160506093425276e+00 ]f64
      y_new = ~V[ -1.452959132853812 -2.187778875125423 ]f64

      k_vals = ~M[
        -2.160506093425276  -2.415858015466959  -2.525217131637079  -2.530906930089893  -2.373278736970216  -2.143782883869835  -2.187778875125423
        -2.172984510849814  -2.034431603317282  -1.715883769683796   2.345467244704591   3.812328420909734   4.768800180323954   3.883778892097804
      ]f64

      y_vals = Nx.stack([y_old, y_new]) |> Nx.transpose()
      t_vals = Nx.tensor(t_old, t_new)

      # tnew = fzero(@(t2) evtfcn_val (evtfcn, t2, ...
      # runge_kutta_interpolate (order, tvals, yvals, ...
      # t2, k_vals), idx2), tvals, optimset ("TolX", 0));

      zero_fn = fn ->
        nil
      end

      t_zero = EqnZeroSolver.find(zero_fn, t_old, t_new)

      expected_t_zero = 2.161317515510217
      assert_in_delta(t_zero, expected_t_zero, 1.0e-02)

      #  call some function here

      t_zero = 2.161317515510217
      y_zero = ~V[ 2.473525941362742e-15  -2.173424479824061e+00 ]f64
    end

    @tag :skip
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

      x = EqnZeroSolver.find(&Math.sin/1, x0, x1)

      expected_x = 3.141592653589795
      assert_in_delta(x, expected_x, 1.0e-02)
    end
  end

  describe "converged?" do
    # From Octave for:
    # fun = @sin
    # x = fzero(fun, [3, 4])

    test "returns :continue if not yet converged" do
      z = %EqnZeroSolver{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      assert EqnZeroSolver.converged?(z, machine_epsilon, tolerance) == :continue
    end

    test "returns :halt if converged" do
      z = %EqnZeroSolver{
        a: 3.141592653589793,
        b: 3.141592653589795,
        u: 3.141592653589793
      }

      machine_epsilon = 2.220446049250313e-16
      tolerance = 2.220446049250313e-16

      assert EqnZeroSolver.converged?(z, machine_epsilon, tolerance) == :halt
    end
  end

  describe "secant" do
    test "works" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %EqnZeroSolver{
        a: 3,
        b: 4,
        u: 3,
        #
        fa: 0.141120008059867,
        fb: -0.756802495307928,
        fu: 0.141120008059867
      }

      c = EqnZeroSolver.interpolate(z, :secant)

      assert_in_delta(c, 3.157162792479947, 1.0e-15)
    end
  end

  describe "bisect" do
    test "works" do
      z = %EqnZeroSolver{a: 3, b: 4}
      assert EqnZeroSolver.interpolate(z, :bisect) == 3.5
    end
  end

  describe "double_secant" do
    test "works" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %EqnZeroSolver{
        a: 3.141592614571824,
        b: 3.157162792479947,
        u: 3.141592614571824,
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fu: 3.901796897832363e-08
      }

      c = EqnZeroSolver.interpolate(z, :double_secant)

      assert_in_delta(c, 3.141592692610915, 1.0e-12)
    end
  end

  describe "too_far?/1" do
    test "returns true if too far" do
      z = %EqnZeroSolver{
        a: 3.2,
        b: 3.4,
        c: 3.0,
        u: 4.0
      }

      assert EqnZeroSolver.too_far?(z) == true
    end

    test "returns false if not too far" do
      z = %EqnZeroSolver{
        a: 3.141592614571824,
        b: 3.157162792479947,
        c: 3.141592692610915,
        u: 3.141592614571824
      }

      assert EqnZeroSolver.too_far?(z) == false
    end
  end

  describe "inverse_cubic_interpolation" do
    test "works" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %EqnZeroSolver{
        a: 3.141281736699444,
        b: 3.157162792479947,
        d: 3.0,
        e: 4.0,
        fa: 3.109168853400020e-04,
        fb: -1.556950978832860e-02,
        fd: 0.141120008059867,
        fe: -0.756802495307928
      }

      c = EqnZeroSolver.interpolate(z, :inverse_cubic_interpolation)
      assert_in_delta(c, 3.141592614571824, 1.0e-12)
    end
  end

  describe "quadratic_interpolation_plus_newton" do
    test "works" do
      # From Octave for:
      # fun = @sin
      # x = fzero(fun, [3, 4])

      z = %EqnZeroSolver{
        a: 3,
        b: 3.157162792479947,
        d: 4,
        fa: 0.141120008059867,
        fb: -1.556950978832860e-02,
        fd: -0.756802495307928,
        fe: 0.141120008059867,
        itype: 2
      }

      c = EqnZeroSolver.interpolate(z, :quadratic_interpolation_plus_newton)

      assert_in_delta(c, 3.141281736699444, 1.0e-15)
    end
  end

  describe "check_for_nonmonotonicity/1" do
    test "monotonic" do
      z = %EqnZeroSolver{
        d: 3.141281736699444,
        fa: 3.901796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08,
        fd: 3.109168853400020e-04
      }

      z = EqnZeroSolver.check_for_nonmonotonicity(z)
      assert_in_delta(z.e, 3.141281736699444, 1.0e-12)
      assert_in_delta(z.fe, 3.109168853400020e-04, 1.0e-12)
    end

    test "non-monotonic" do
      z = %EqnZeroSolver{
        d: 3.141281736699444,
        fa: -3.911796897832363e-08,
        fb: -1.556950978832860e-02,
        fc: -3.902112221087341e-08,
        fd: 3.109168853400020e-04
      }

      z = EqnZeroSolver.check_for_nonmonotonicity(z)
      assert_in_delta(z.fe, -3.902112221087341e-08, 1.0e-12)
    end
  end

  describe "compute_new_point" do
    test "works" do
      z = %EqnZeroSolver{
        c: 3.141281736699444,
        iteration_count: 1,
        fn_eval_count: 3,
        fc: 7
      }

      zero_fn = &Math.sin/1
      z = EqnZeroSolver.compute_new_point(z, zero_fn)

      assert_in_delta(z.fc, 3.109168853400020e-04, 1.0e-16)
      assert z.iteration_count == 2
      assert z.fn_eval_count == 4
    end
  end
end
