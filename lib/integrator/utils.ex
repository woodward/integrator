defmodule Integrator.Utils do
  @moduledoc """
  Various utility functions used in `Integrator`
  """
  import Nx.Defn

  @doc """
  Performs a 3rd order Hermite interpolation. Adapted from function `hermite_cubic_interpolation` in
  [runge_kutta_interpolate.m](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_interpolate.m)


  See [Wikipedia](https://en.wikipedia.org/wiki/Cubic_Hermite_spline)
  """
  @spec hermite_cubic_interpolation(Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
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

  @coefs_u_half [
    6_025_192_743 / 30_085_553_152,
    0.0,
    51_252_292_925 / 65_400_821_598,
    -2_691_868_925 / 45_128_329_728,
    187_940_372_067 / 1_594_534_317_056,
    -1_776_094_331 / 19_743_644_256,
    11_237_099 / 235_043_384
  ]

  @doc """
  Performs a 4th order Hermite interpolation. Used by an ODE solver to interpolate the
  solution at the time `t_out`. As proposed by Shampine in Lawrence, Shampine,
  "Some Practical Runge-Kutta Formulas", 1986.

  See [hermite_quartic_interpolation function in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_interpolate.m#L91).
  """
  @spec hermite_quartic_interpolation(Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn hermite_quartic_interpolation(t, x, der, t_out) do
    dt = t[1] - t[0]
    x_col1 = Nx.slice_along_axis(x, 0, 1, axis: 1)

    # 4th order approximation of x in t+dt/2 as proposed by Shampine in
    # Lawrence, Shampine, "Some Practical Runge-Kutta Formulas", 1986.
    u_half = x_col1 + 0.5 * dt * Nx.new_axis(Nx.dot(der, Nx.tensor(@coefs_u_half, type: Nx.type(x))), 1)

    # Rescale time on [0,1]
    s = (t_out - t[0]) / dt

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    # Hermite basis functions

    # H0 = x1 = 1   - 11*s^2 + 18*s^3 -  8*s^4
    # H1 = x2 =   s -  4*s^2 +  5*s^3 -  2*s^4
    # H2 = x3 =       16*s^2 - 32*s^3 + 16*s^4
    # H3 = x4 =     -  5*s^2 + 14*s^3 -  8*s^4
    # H4 = x5 =          s^2 -  3*s^3 +  2*s^4

    x1 = (1.0 - 11.0 * s2 + 18.0 * s3 - 8.0 * s4) * x_col1

    der_col_1 = Nx.slice_along_axis(der, 0, 1, axis: 1)
    x2 = (s - 4.0 * s2 + 5.0 * s3 - 2.0 * s4) * (dt * der_col_1)

    x3 = (16.0 * s2 - 32.0 * s3 + 16.0 * s4) * u_half

    x_col2 = Nx.slice_along_axis(x, 1, 1, axis: 1)
    x4 = (-5.0 * s2 + 14.0 * s3 - 8.0 * s4) * x_col2

    # Note that we are assuming that "der" has 7 columns here:
    der_last_col = Nx.slice_along_axis(der, 6, 1, axis: 1)
    x5 = (s2 - 3.0 * s3 + 2.0 * s4) * (dt * der_last_col)

    x1 + x2 + x3 + x4 + x5
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
  reuse it in subsequent calls to `kahan_sum/3` function.

  The second input argument contains the compensation term and is returned
  as the second output argument so that it can be reused in future calls of
  the same summation.

  The third input argument `term` is the variable to be added to `sum`.
  """
  @spec kahan_sum(Nx.t(), Nx.t(), Nx.t()) :: {Nx.t(), Nx.t()}
  defn kahan_sum(sum, comp, term) do
    # Octave code:
    #   x = term - comp;
    #   t = sum + x;
    #   comp = (t - sum) - x;
    #   sum = t;

    x = term - comp
    t = sum + x

    {t, t - sum - x}
  end

  @doc """
  Returns the sign of the tensor as -1 or 1 (or 0 for zero tensors)
  """
  @spec sign(float()) :: float()
  def sign(x) when x < 0.0, do: -1.0
  def sign(x) when x > 0.0, do: 1.0
  def sign(_x), do: 0.0

  @doc """
  Returns the columns of a tensor as a list of vector tensors

  E.g., a tensor of the form:

       ~MAT[
        1  2  3  4
        5  6  7  8
        9 10 11 12
      ]s8

  will return

    [
      ~VEC[ 1  5   9 ]s8,
      ~VEC[ 2  6  10 ]s8,
      ~VEC[ 3  7  11 ]s8,
      ~VEC[ 4  8  12 ]s8
    ]

  """
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

  @doc """
  Converts a Nx vector into a list of 1-D tensors

  Is there an existing Nx way to do this?  If so, swap the usage of this function
  and then delete this

  Note that

      vector |> Nx.as_list() |> Enum.map(& Nx.tensor(&1, type: Nx.type(vector)))

  seems to introduce potential precision issues
  """
  @spec vector_as_list(Nx.t()) :: [Nx.t()]
  def vector_as_list(vector) do
    {length} = Nx.shape(vector)

    0..(length - 1)
    |> Enum.reduce([], fn i, acc ->
      [vector[i] | acc]
    end)
    |> Enum.reverse()
  end

  @doc """
  Given a list of elements, create a new list with only the unique elements from the list
  """
  @spec unique(list()) :: list()
  def unique(values) do
    MapSet.new(values) |> MapSet.to_list() |> Enum.sort()
  end
end
