defmodule Integrator.Utils do
  @moduledoc """
  Various utility functions used in `Integrator`
  """
  import Nx.Defn

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

  # Perhaps this can be deleted soon?
  def tensor_length(tensor) do
    case Nx.shape(tensor) do
      {} -> 1
      {length} -> length
    end
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
end
