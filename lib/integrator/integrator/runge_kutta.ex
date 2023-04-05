defmodule Integrator.RungeKutta do
  @moduledoc false

  import Nx.Defn

  @a Nx.tensor(
       [
         [0, 0, 0, 0, 0, 0],
         [1 / 5, 0, 0, 0, 0, 0],
         [3 / 40, 9 / 40, 0, 0, 0, 0],
         [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
         [19_372 / 6_561, -25_360 / 2187, 64_448 / 6561, -212 / 729, 0, 0],
         [9_017 / 3_168, -355 / 33, 46_732 / 5247, 49 / 176, -5103 / 18_656, 0]
       ],
       type: :f64
     )

  @b Nx.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], type: :f64)
  @c Nx.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], type: :f64)
  @c_prime Nx.tensor([5179 / 57_600, 0, 7571 / 16_695, 393 / 640, -92_097 / 339_200, 187 / 2100, 1 / 40], type: :f64)

  defn dormand_prince_45(ode_fn, t, x, dt, k_vals) do
    t_next = t + dt
    s = t + dt * @b
    cc = dt * @c
    aa = dt * @a
    # k = zeros (rows (x), 7);
    {length_of_x} = Nx.shape(x)
    # k = Nx.broadcast(0.0, {length_of_x, 7})

    # Do an if statement here in the future for when we do NOT use the last k_vals for the first evaluation
    # i.e., it becomes another function evaluation
    k0 = Nx.slice_along_axis(k_vals, 6, length_of_x - 1, axis: 1) |> Nx.flatten()
    # k0 checks out

    # Octave code:
    # k(:,2) = feval (fcn, s(2), x + k(:,1)   * aa(2, 1).'  , args{:});
    # k(:,3) = feval (fcn, s(3), x + k(:,1:2) * aa(3, 1:2).', args{:});
    # k(:,4) = feval (fcn, s(4), x + k(:,1:3) * aa(4, 1:3).', args{:});
    # k(:,5) = feval (fcn, s(5), x + k(:,1:4) * aa(5, 1:4).', args{:});
    # k(:,6) = feval (fcn, s(6), x + k(:,1:5) * aa(6, 1:5).', args{:});

    # Octave code with indices converted to zero based:
    # k(:,1) = feval (fcn, s(1), x + k(:,0)   * aa(1, 0).'  , args{:});
    # k(:,2) = feval (fcn, s(2), x + k(:,0:1) * aa(2, 0:1).', args{:});
    # k(:,3) = feval (fcn, s(3), x + k(:,0:2) * aa(3, 0:2).', args{:});
    # k(:,4) = feval (fcn, s(4), x + k(:,0:3) * aa(4, 0:3).', args{:});
    # k(:,5) = feval (fcn, s(5), x + k(:,0:4) * aa(5, 0:4).', args{:});

    aa_1 = aa[1][0]
    aa_2 = Nx.stack([aa[2][0], aa[2][1]])
    aa_3 = Nx.stack([aa[3][0], aa[3][1], aa[3][2]])
    aa_4 = Nx.stack([aa[4][0], aa[4][1], aa[4][2], aa[4][3]])
    aa_5 = Nx.stack([aa[5][0], aa[5][1], aa[5][2], aa[5][3], aa[5][4]])

    # k1 checks out
    k1 = ode_fn.(s[1], x + k0 * aa_1)
    k_0_1 = Nx.stack([k0, k1])
    k_0_1_t = k_0_1 |> Nx.transpose()

    k2 = ode_fn.(s[2], x + Nx.dot(k_0_1_t, Nx.transpose(aa_2)))
    k_0_2 = Nx.stack([k0, k1, k2])
    k_0_2_t = k_0_2 |> Nx.transpose()

    k3 = ode_fn.(s[3], x + Nx.dot(k_0_2_t, Nx.transpose(aa_3)))
    k_0_3 = Nx.stack([k0, k1, k2, k3])
    k_0_3_t = k_0_3 |> Nx.transpose()

    k4 = ode_fn.(s[4], x + Nx.dot(k_0_3_t, Nx.transpose(aa_4)))
    k_0_4 = Nx.stack([k0, k1, k2, k3, k4])
    k_0_4_t = k_0_4 |> Nx.transpose()

    k5 = ode_fn.(s[5], x + Nx.dot(k_0_4_t, Nx.transpose(aa_5)))

    k_0_5 = Nx.stack([k0, k1, k2, k3, k4, k5])
    k_0_5_t = k_0_5 |> Nx.transpose()
    x_next = x + Nx.dot(k_0_5_t, cc)

    k6 = ode_fn.(t_next, x_next)
    k_new = Nx.stack([k0, k1, k2, k3, k4, k5, k6]) |> Nx.transpose()
    cc_prime = dt * @c_prime
    x_est = x + Nx.dot(k_new, cc_prime)

    {t_next, x_next, x_est, k_new}
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
