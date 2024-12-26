defmodule Integrator.NonLinearEqnRootRefactorPrivateTest do
  # Tests of private functions using Patch in NonLinearEqnRoot

  @moduledoc false
  use Integrator.TestCase, async: false
  alias Integrator.NonLinearEqnRootRefactor

  describe "converged?" do
    # From Octave for:
    # fun = @sin
    # x = fzero(fun, [3, 4])

    @tag transferred_to_refactor?: false
    test "returns :continue (i.e., 0) if not yet converged" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141592614571824, type: :f64),
        b: Nx.tensor(3.157162792479947, type: :f64),
        u: Nx.tensor(3.141592614571824, type: :f64)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      assert NonLinearEqnRootRefactor.converged?(z, machine_epsilon, tolerance) == Nx.tensor(0)
    end

    @tag transferred_to_refactor?: false
    test "returns :halt (i.e., 1) if converged" do
      z = %NonLinearEqnRootRefactor{
        a: Nx.tensor(3.141592653589793, type: :f64),
        b: Nx.tensor(3.141592653589795, type: :f64),
        u: Nx.tensor(3.141592653589793, type: :f64)
      }

      machine_epsilon = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)

      assert NonLinearEqnRootRefactor.converged?(z, machine_epsilon, tolerance) == Nx.tensor(1)
    end
  end
end
