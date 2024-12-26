defmodule Integrator.NonLinearEqnRoot.InternalComputations do
  @moduledoc false

  import Nx.Defn
  alias Integrator.NonLinearEqnRootRefactor

  @spec converged?(NonLinearEqnRootRefactor.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn converged?(z, machine_eps, tolerance) do
    z.b - z.a <= 2 * (2 * Nx.abs(z.u) * machine_eps + tolerance)
  end

  @spec interpolate_secant(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_secant(z) do
    z.u - (z.a - z.b) / (z.fa - z.fb) * z.fu
  end
end
