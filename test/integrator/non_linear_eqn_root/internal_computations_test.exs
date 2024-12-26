defmodule Integrator.NonLinearEqnRoot.InternalComputationsTest do
  # Tests of private functions using Patch in NonLinearEqnRoot

  @moduledoc false
  use Integrator.TestCase, async: false
  alias Integrator.NonLinearEqnRootRefactor
  alias Integrator.NonLinearEqnRoot.InternalComputations

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

      c = InternalComputations.interpolate_quadratic_interpolation_plus_newton(z)

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

      c = InternalComputations.interpolate_quadratic_interpolation_plus_newton(z)

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

      c = InternalComputations.interpolate_inverse_cubic_interpolation(z)
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
end
