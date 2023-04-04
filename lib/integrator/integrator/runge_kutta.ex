defmodule Integrator.RungeKutta do
  @moduledoc false

  import Nx.Defn

  defn dormand_prince_45(ode_fn, t, x, dt, k_vals, opts \\ []) do
    t_next = t + dt
    {t_next}
  end

  @coefs_u_half Nx.tensor(
                  [
                    6_025_192_743 / 30_085_553_152,
                    0.0,
                    51_252_292_925 / 65_400_821_598,
                    -2_691_868_925 / 45_128_329_728,
                    187_940_372_067 / 1_594_534_317_056,
                    -1_776_094_331 / 19_743_644_256,
                    11_237_099 / 235_043_384
                  ],
                  type: :f64
                )

  @doc """
  Performs a 4th order Hermite interpolation. Used by an ODE solver to interpolate the
  solution at the time `t_out`.

  See [code in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_interpolate.m#L91).
  """
  defn hermite_quartic_interpolation(t, x, der, t_out) do
    # Octave code:
    #   persistent coefs_u_half = ...
    #   [6025192743/30085553152; 0; 51252292925/65400821598;
    #    -2691868925/45128329728; 187940372067/1594534317056;
    #    -1776094331/19743644256; 11237099/235043384];

    # ## 4th order approximation of y in t+dt/2 as proposed by Shampine in
    # ## Lawrence, Shampine, "Some Practical Runge-Kutta Formulas", 1986.
    # dt = t(2) - t(1);
    # u_half = x(:,1) + (1/2) * dt * (der(:,1:7) * coefs_u_half);

    # ## Rescale time on [0,1]
    # s = (t_out - t(1)) / dt;

    # ## Hermite basis functions
    # ## H0 = 1   - 11*s.^2 + 18*s.^3 -  8*s.^4;
    # ## H1 =   s -  4*s.^2 +  5*s.^3 -  2*s.^4;
    # ## H2 =       16*s.^2 - 32*s.^3 + 16*s.^4;
    # ## H3 =     -  5*s.^2 + 14*s.^3 -  8*s.^4;
    # ## H4 =          s.^2 -  3*s.^3 +  2*s.^4;

    # x_out = (1   - 11*s.^2 + 18*s.^3 -  8*s.^4) .* x(:,1) + ...
    #         (  s -  4*s.^2 +  5*s.^3 -  2*s.^4) .* (dt * der(:,1)) + ...
    #         (      16*s.^2 - 32*s.^3 + 16*s.^4) .* u_half + ...
    #         (    -  5*s.^2 + 14*s.^3 -  8*s.^4) .* x(:,2) + ...
    #         (         s.^2 -  3*s.^3 +  2*s.^4) .* (dt * der(:,end));

    dt = t[1] - t[0]

    x_col1 = Nx.slice_along_axis(x, 0, 1, axis: 1)

    u_half = x_col1 + 0.5 * dt * Nx.new_axis(Nx.dot(der, @coefs_u_half), 1)

    s = (t_out - t[0]) / dt
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    x1 = (1.0 - 11.0 * s2 + 18.0 * s3 - 8.0 * s4) * x_col1

    der_col_1 = Nx.slice_along_axis(der, 0, 1, axis: 1)
    x2 = (s - 4.0 * s2 + 5.0 * s3 - 2.0 * s4) * (dt * der_col_1)

    x3 = (16.0 * s2 - 32.0 * s3 + 16.0 * s4) * u_half

    x_col2 = Nx.slice_along_axis(x, 1, 1, axis: 1)
    x4 = (-5.0 * s2 + 14.0 * s3 - 8.0 * s4) * x_col2

    der_last_col = Nx.slice_along_axis(der, 6, 1, axis: 1)
    x5 = (s2 - 3.0 * s3 + 2.0 * s4) * (dt * der_last_col)

    x1 + x2 + x3 + x4 + x5
  end
end
