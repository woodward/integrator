defmodule Integrator.Utils do
  @moduledoc false
  import Nx.Defn

  @default_abs_tol 1.0e-06
  @default_rel_tol 1.0e-03
  @default_norm_control true

  def default_opts() do
    [abs_tol: @default_abs_tol, rel_tol: @default_rel_tol, norm_control: @default_norm_control]
  end

  @doc """
  Based on
  [Octave function AbsRelNorm](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/AbsRel_norm.m)

  ## Options
  * `:norm_control` - Control error relative to norm; i.e., control the error `e` at each step using the norm of the
    solution rather than its absolute value.  Defaults to true.

    See [Matlab documentation](https://www.mathworks.com/help/matlab/ref/odeset.html#bu2m9z6-NormControl)
    for a description of norm control.
  """
  defn abs_rel_norm(x, x_old, y, abs_tolerance, rel_tolerance, opts \\ []) do
    opts = keyword!(opts, norm_control: @default_norm_control, abs_tol: @default_abs_tol, rel_tol: @default_rel_tol)

    if opts[:norm_control] do
      # Octave code
      # sc = max (AbsTol(:), RelTol * max (sqrt (sumsq (x)), sqrt (sumsq (x_old))));
      # retval = sqrt (sumsq ((x - y))) / sc;

      max_sq_x = Nx.max(sum_sq(x), sum_sq(x_old))
      sc = Nx.max(abs_tolerance, rel_tolerance * max_sq_x)
      sum_sq(x - y) / sc
    else
      # Octave code:
      # sc = max (AbsTol(:), RelTol .* max (abs (x), abs (x_old)));
      # retval = max (abs (x - y) ./ sc);

      sc = Nx.max(abs_tolerance, rel_tolerance * Nx.max(Nx.abs(x), Nx.abs(x_old)))
      (Nx.abs(x - y) / sc) |> Nx.reduce_max()
    end
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
  solution at the time `t_out`. As proposed by Shampine in Lawrence, Shampine,
  "Some Practical Runge-Kutta Formulas", 1986.

  See [code in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_interpolate.m#L91).
  """
  defn hermite_quartic_interpolation(t, x, der, t_out) do
    # Octave code:
    #   persistent coefs_u_half = ...
    #   [6025192743/30085553152; 0; 51252292925/65400821598;
    #    -2691868925/45128329728; 187940372067/1594534317056;
    #    -1776094331/19743644256; 11237099/235043384];

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

  @doc """
  Computes a good initial timestep for an ODE solver of order `order`
  using the algorithm described in the reference below.

  The input argument `ode_fn`, is the function describing the differential
  equations, `t0` is the initial time, and `x0` is the initial
  condition.  `abs_tol` and `rel_tol` are the absolute and relative
  tolerance on the ODE integration.

  Reference:

  E. Hairer, S.P. Norsett and G. Wanner,
  "Solving Ordinary Differential Equations I: Nonstiff Problems",
  Springer.
  """
  defn starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, opts \\ []) do
    # Compute norm of initial conditions
    y_zeros = zero_vector(x0)
    d0 = abs_rel_norm(x0, x0, y_zeros, abs_tol, rel_tol, opts)

    y = ode_fn.(t0, x0)

    d1 = abs_rel_norm(y, y, y_zeros, abs_tol, rel_tol, opts)

    h0 =
      if d0 < 1.0e-5 or d1 < 1.0e-5 do
        1.0e-6
      else
        0.01 * (d0 / d1)
      end

    # Compute one step of Explicit-Euler
    x1 = x0 + h0 * y

    # Approximate the derivative norm
    yh = ode_fn.(t0 + h0, x1)

    d2 = 1.0 / h0 * abs_rel_norm(yh - y, yh - y, y_zeros, abs_tol, rel_tol, opts)

    h1 =
      if Nx.max(d1, d2) <= 1.0e-15 do
        Nx.max(1.0e-6, h0 * 1.0e-3)
      else
        Nx.pow(1.0e-2 / Nx.max(d1, d2), 1 / (order + 1))
      end

    Nx.min(100.0 * h0, h1)
  end

  defnp sum_sq(x) do
    (x * x) |> Nx.sum() |> Nx.sqrt()
  end

  @doc """
  Creates a zero vector that has the length of `x`
  """
  defn zero_vector(x) do
    {length_of_x} = Nx.shape(x)
    Nx.broadcast(0.0, {length_of_x})
  end

  def sign(x) when x < 0.0, do: -1.0
  def sign(x) when x > 0.0, do: 1.0
  def sign(_x), do: 0.0

  def columns_as_list(matrix, start_index, end_index \\ nil) do
    matrix_t = Nx.transpose(matrix)

    end_index =
      if end_index do
        end_index
      else
        {_n_rows, n_cols} = Nx.shape(matrix)
        n_cols - 1
      end

    start_index..end_index
    |> Enum.reduce([], fn i, acc ->
      col = Nx.slice_along_axis(matrix_t, i, 1, axis: 0) |> Nx.flatten()
      [col | acc]
    end)
    |> Enum.reverse()
  end

  # In Octave, get these via eps("single") or eps("double")
  @epislon_f32 1.1920929e-07
  @epislon_f64 2.220446049250313e-16

  def epsilon(:f32), do: @epislon_f32
  def epsilon({:f, 32}), do: @epislon_f32

  def epsilon(:f64), do: @epislon_f64
  def epsilon({:f, 64}), do: @epislon_f64

  def unique(values) do
    MapSet.new(values) |> MapSet.to_list() |> Enum.sort()
  end
end
