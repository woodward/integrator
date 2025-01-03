defmodule Integrator.Utils do
  @moduledoc """
  Various utility functions used in `Integrator`
  """
  import Nx.Defn
  import Nx, only: [sign: 1]

  alias Integrator.TensorTypeError

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

  # Perhaps this can be deleted soon?
  def tensor_length(tensor) do
    case Nx.shape(tensor) do
      {} -> 1
      {length} -> length
    end
  end

  # Paulo said these sign functions might have numerical issues in the Octave version so do this instead

  @doc """
  Returns true if both quantities have the same sign
  """
  @spec same_signs?(Nx.t(), Nx.t()) :: Nx.t()
  defn same_signs?(x1, x2) do
    # In original Octave as the following; uncomment to verify working correctly:
    # sign(x1) * sign(x2) > 0

    sign_x1 = sign(x1)
    sign_x2 = sign(x2)

    both_signs_are_zero = sign_x1 == 0 and sign_x2 == 0
    sign_x1 == sign_x2 and not both_signs_are_zero
  end

  @doc """
  Returns true if both quantities have the same sign, or if one or more of them is zero
  """
  @spec same_signs_or_any_zeros?(Nx.t(), Nx.t()) :: Nx.t()
  defn same_signs_or_any_zeros?(x1, x2) do
    # In original Octave as the following; uncomment to verify working correctly:
    # sign(x1) * sign(x2) >= 0

    sign_x1 = sign(x1)
    sign_x2 = sign(x2)

    sign_x1 == sign_x2 or sign_x1 == 0 or sign_x2 == 0
  end

  @doc """
  Returns true if both quantities have different signs, or if one or more of them is zero
  """
  @spec different_signs_or_any_zeros?(Nx.t(), Nx.t()) :: Nx.t()
  defn different_signs_or_any_zeros?(x1, x2) do
    # In original Octave as the following; uncomment to verify working correctly:
    # sign(x1) * sign(x2) <= 0

    sign_x1 = sign(x1)
    sign_x2 = sign(x2)

    sign_x1 != sign_x2 or sign_x1 == 0 or sign_x2 == 0
  end

  @doc """
  Returns true if both quantities have different signs (and neither one of them is zero)
  """
  @spec different_signs?(Nx.t(), Nx.t()) :: Nx.t()
  defn different_signs?(x1, x2) do
    # In original Octave as the following; uncomment to verify working correctly:
    # sign(x1) * sign(x2) < 0

    sign_x1 = sign(x1)
    sign_x2 = sign(x2)

    either_sign_is_zero = sign_x1 == 0 or sign_x2 == 0
    sign_x1 != sign_x2 and not either_sign_is_zero
  end

  @doc """
  Returns the elapsed time (in microseconds) given a starting timestamp (also in microseconds)
  """
  @spec elapsed_time_μs(Nx.t()) :: Nx.t()
  def elapsed_time_μs(start_time_μs), do: Nx.subtract(Nx.s32(:os.system_time(:microsecond)), start_time_μs)

  @doc """
  Returns the timestamp in microseconds as an Nx :s32 tensor
  """
  @spec timestamp_μs() :: Nx.t()
  def timestamp_μs, do: Nx.s32(:os.system_time(:microsecond))

  @doc """
  A function which converts args to their Nx equivalents. Used to populate Nx.Container structs with
  option values than safely cross the Elixir/Nx boundary safely and also be of known, expected types.
  """
  @spec convert_arg_to_nx_type(Nx.Tensor.t() | float() | integer() | fun(), Nx.Type.t()) :: Nx.t()
  def convert_arg_to_nx_type(%Nx.Tensor{} = arg, type) do
    if Nx.type(arg) != type, do: raise(TensorTypeError)
    arg
  end

  def convert_arg_to_nx_type(arg, _type) when is_function(arg), do: arg
  def convert_arg_to_nx_type(arg, type), do: Nx.tensor(arg, type: type)
end
