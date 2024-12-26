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
end
