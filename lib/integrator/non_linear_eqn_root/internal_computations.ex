defmodule Integrator.NonLinearEqnRoot.InternalComputations do
  @moduledoc false

  import Nx.Defn
  alias Integrator.NonLinearEqnRootRefactor

  @spec converged?(NonLinearEqnRootRefactor.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn converged?(z, machine_eps, tolerance) do
    z.b - z.a <= 2 * (2 * Nx.abs(z.u) * machine_eps + tolerance)
  end

  @spec too_far?(Nx.t(), NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn too_far?(c, z) do
    Nx.abs(c - z.u) > 0.5 * (z.b - z.a)
  end

  # Modification 2: skip inverse cubic interpolation if nonmonotonicity is detected
  @spec check_for_non_monotonicity(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn check_for_non_monotonicity(z) do
    if Nx.sign(z.fc - z.fa) * Nx.sign(z.fc - z.fb) >= 0 do
      # The new point broke monotonicity.
      # Disable inverse cubic:
      %{z | fe: z.fc}
    else
      %{z | e: z.d, fe: z.fd}
    end
  end

  @spec interpolate_quadratic_interpolation_plus_newton(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_quadratic_interpolation_plus_newton(z) do
    a0 = z.fa
    a1 = (z.fb - z.fa) / (z.b - z.a)
    a2 = ((z.fd - z.fb) / (z.d - z.b) - a1) / (z.d - z.a)

    ## Modification 1: this is simpler and does not seem to be worse.
    c = z.a - a0 / a1

    if a2 != 0 do
      {_z, _a0, _a1, _a2, c, _i} =
        while {z, a0, a1, a2, c, i = 1}, Nx.less_equal(i, z.iter_type) do
          pc = a0 + (a1 + a2 * (c - z.b)) * (c - z.a)
          pdc = a1 + a2 * (2 * c - z.a - z.b)

          new_c =
            if pdc == 0 do
              # Octave does a break here - is the c = 0 caught downstream? Need to handle this case somehow"
              # Note that there is NO test case for this case, as I couldn't figure out how to set up
              # the initial conditions to reach here
              z.a - a0 / a1
            else
              c - pc / pdc
            end

          {z, a0, a1, a2, new_c, i + 1}
        end

      c
    else
      c
    end
  end

  @spec interpolate_inverse_cubic_interpolation(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_inverse_cubic_interpolation(z) do
    q11 = (z.d - z.e) * z.fd / (z.fe - z.fd)
    q21 = (z.b - z.d) * z.fb / (z.fd - z.fb)
    q31 = (z.a - z.b) * z.fa / (z.fb - z.fa)
    d21 = (z.b - z.d) * z.fd / (z.fd - z.fb)
    d31 = (z.a - z.b) * z.fb / (z.fb - z.fa)

    q22 = (d21 - q11) * z.fb / (z.fe - z.fb)
    q32 = (d31 - q21) * z.fa / (z.fd - z.fa)
    d32 = (d31 - q21) * z.fd / (z.fd - z.fa)
    q33 = (d32 - q22) * z.fa / (z.fe - z.fa)

    z.a + q31 + q32 + q33
  end

  @spec interpolate_double_secant(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_double_secant(z) do
    z.u - 2.0 * (z.b - z.a) / (z.fb - z.fa) * z.fu
  end

  @spec interpolate_bisect(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_bisect(z) do
    0.5 * (z.b + z.a)
  end

  @spec interpolate_secant(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_secant(z) do
    z.u - (z.a - z.b) / (z.fa - z.fb) * z.fu
  end
end
