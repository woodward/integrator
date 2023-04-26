defmodule Integrator.Utils do
  @moduledoc """
  Various utility functions used in `Integrator`
  """
  import Nx.Defn

  @default_norm_control true

  @default_opts [
    abs_tol: 1.0e-06,
    rel_tol: 1.0e-03,
    norm_control: @default_norm_control
  ]

  @doc """
  Gets the default options for the functions in `Integrator.Utils`
  """
  @spec default_opts() :: Keyword.t()
  def default_opts(), do: @default_opts

  @doc """
  Originally based on
  [Octave function AbsRelNorm](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/AbsRel_norm.m)

  ## Options
  * `:norm_control` - Control error relative to norm; i.e., control the error `e` at each step using the norm of the
    solution rather than its absolute value.  Defaults to true.

    See [Matlab documentation](https://www.mathworks.com/help/matlab/ref/odeset.html#bu2m9z6-NormControl)
    for a description of norm control.
  """
  @spec abs_rel_norm(Nx.t(), Nx.t(), Nx.t(), float(), float(), Keyword.t()) :: Nx.t()
  defn abs_rel_norm(t, t_old, x, abs_tolerance, rel_tolerance, opts \\ []) do
    opts = keyword!(opts, norm_control: @default_norm_control)

    if opts[:norm_control] do
      # Octave code
      # sc = max (AbsTol(:), RelTol * max (sqrt (sumsq (t)), sqrt (sumsq (t_old))));
      # retval = sqrt (sumsq ((t - x))) / sc;

      max_sq_t = Nx.max(sum_sq(t), sum_sq(t_old))
      sc = Nx.max(abs_tolerance, rel_tolerance * max_sq_t)
      sum_sq(t - x) / sc
    else
      # Octave code:
      # sc = max (AbsTol(:), RelTol .* max (abs (t), abs (t_old)));
      # retval = max (abs (t - x) ./ sc);

      sc = Nx.max(abs_tolerance, rel_tolerance * Nx.max(Nx.abs(t), Nx.abs(t_old)))
      (Nx.abs(t - x) / sc) |> Nx.reduce_max()
    end
  end

  @doc """
  Performs a 3rd order Hermite interpolation. Adapted from function `hermite_cubic_interpolation` in
  [runge_kutta_interpolate.m](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_interpolate.m)


  See [Wikipedia](https://en.wikipedia.org/wiki/Cubic_Hermite_spline)
  """
  @spec hermite_cubic_interpolation(float() | Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn hermite_cubic_interpolation(t, x, der, t_out) do
    # Octave:
    #   dt = (t(2) - t(1));
    #   s = (t_out - t(1)) / dt;
    #   x_out = ((1 + 2*s) .* (1-s).^2) .* x(:,1) + ...
    #           (s .* (1-s).^2 * dt   ) .* der(:,1) + ...
    #           ((3-2*s) .* s.^2      ) .* x(:,end) + ...
    #           ((s-1) .* s.^2   * dt ) .* der(:,end);

    dt = t[1] - t[0]
    s = (t_out - t[0]) / dt

    x_col1 = Nx.slice_along_axis(x, 0, 1, axis: 1)
    der_col_1 = Nx.slice_along_axis(der, 0, 1, axis: 1)
    x_col2 = Nx.slice_along_axis(x, 1, 1, axis: 1)
    # Note that we are assuming "der" has 4 columns:
    der_last_col = Nx.slice_along_axis(der, 3, 1, axis: 1)

    s_minus_1 = 1 - s
    s_minus_1_sq = s_minus_1 * s_minus_1

    x1 = (1 + 2 * s) * s_minus_1_sq * x_col1
    x2 = s * s_minus_1_sq * dt * der_col_1
    x3 = (3 - 2 * s) * s * s * x_col2
    x4 = (s - 1) * s * s * dt * der_last_col

    x1 + x2 + x3 + x4
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

  See [hermite_quartic_interpolation function in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_interpolate.m#L91).
  """
  @spec hermite_quartic_interpolation(float() | Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn hermite_quartic_interpolation(t, x, der, t_out) do
    # Octave code:
    #   dt = t(2) - t(1);
    #   u_half = x(:,1) + (1/2) * dt * (der(:,1:7) * coefs_u_half);

    #   ## Rescale time on [0,1]
    #   s = (t_out - t(1)) / dt;

    #   ## Hermite basis functions
    #   ## H0 = 1   - 11*s.^2 + 18*s.^3 -  8*s.^4;
    #   ## H1 =   s -  4*s.^2 +  5*s.^3 -  2*s.^4;
    #   ## H2 =       16*s.^2 - 32*s.^3 + 16*s.^4;
    #   ## H3 =     -  5*s.^2 + 14*s.^3 -  8*s.^4;
    #   ## H4 =          s.^2 -  3*s.^3 +  2*s.^4;

    #   x_out = (1   - 11*s.^2 + 18*s.^3 -  8*s.^4) .* x(:,1) + ...
    #           (  s -  4*s.^2 +  5*s.^3 -  2*s.^4) .* (dt * der(:,1)) + ...
    #           (      16*s.^2 - 32*s.^3 + 16*s.^4) .* u_half + ...
    #           (    -  5*s.^2 + 14*s.^3 -  8*s.^4) .* x(:,2) + ...
    #           (         s.^2 -  3*s.^3 +  2*s.^4) .* (dt * der(:,end));

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

    # Note that we are assuming that "der" has 7 columns:
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

  Originally based on [`starting_stepsize.m`](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/starting_stepsize.m).

  Reference:

  E. Hairer, S.P. Norsett and G. Wanner,
  "Solving Ordinary Differential Equations I: Nonstiff Problems",
  Springer.
  """
  @spec starting_stepsize(integer(), fun(), float(), Nx.t(), float(), float(), Keyword.t()) :: Nx.t()
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

  @doc """
  Implements the Kahan summation algorithm, also known as compensated summation.
  Based on this [code in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/kahan.m).
  This is really a private function, but is made public so the docs are visible.

  The algorithm significantly reduces the numerical error in the total
  obtained by adding a sequence of finite precision floating point numbers
  compared to the straightforward approach.  For more details
  see [this Wikipedia entry](http://en.wikipedia.org/wiki/Kahan_summation_algorithm).
  This function is called by AdaptiveStepsize.integrate to better catch
  equality comparisons.

  The first input argument is the variable that will contain the summation.
  This variable is also returned as the first output argument in order to
  reuse it in subsequent calls to `Integrator.AdaptiveStepsize.kahan_sum/3` function.

  The second input argument contains the compensation term and is returned
  as the second output argument so that it can be reused in future calls of
  the same summation.

  The third input argument `term` is the variable to be added to `sum`.
  """
  @spec kahan_sum(Nx.t(), Nx.t(), Nx.t()) :: {Nx.t(), Nx.t()}
  defn kahan_sum(sum, comp, term) do
    # Octave code:
    # x = term - comp;
    # t = sum + x;
    # comp = (t - sum) - x;
    # sum = t;

    x = term - comp
    t = sum + x
    comp = t - sum - x
    sum = t

    {sum, comp}
  end

  defnp sum_sq(x) do
    (x * x) |> Nx.sum() |> Nx.sqrt()
  end

  @doc """
  Creates a zero vector that has the length of `x`
  """
  @spec zero_vector(Nx.t()) :: Nx.t()
  defn zero_vector(x) do
    {length_of_x} = Nx.shape(x)
    Nx.broadcast(0.0, {length_of_x})
  end

  @spec sign(Nx.t()) :: float()
  def sign(x) when x < 0.0, do: -1.0
  def sign(x) when x > 0.0, do: 1.0
  def sign(_x), do: 0.0

  @spec columns_as_list(Nx.t(), integer(), integer() | nil) :: [Nx.t()]
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

  @spec epsilon(Nx.Type.t()) :: float()
  def epsilon(:f32), do: @epislon_f32
  def epsilon({:f, 32}), do: @epislon_f32

  def epsilon(:f64), do: @epislon_f64
  def epsilon({:f, 64}), do: @epislon_f64

  @spec unique(list()) :: list()
  def unique(values) do
    MapSet.new(values) |> MapSet.to_list() |> Enum.sort()
  end

  @spec type_atom(Nx.t()) :: atom()
  def type_atom(tensor) do
    tensor |> Nx.type() |> Nx.Type.to_string() |> String.to_atom()
  end
end
