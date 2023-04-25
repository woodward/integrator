defmodule Integrator.Helpers do
  @moduledoc false
  import ExUnit.Assertions

  @doc """
  Asserts `lhs` is close to `rhs`.
  Copied from [Nx.Helpers](https://github.com/elixir-nx/nx/blob/main/nx/test/support/helpers.ex)
  (which is not included in the released version of Nx, so I cannot just invoke it).
  """
  def assert_all_close(lhs, rhs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    unless Nx.all_close(lhs, rhs, atol: atol, rtol: rtol, equal_nan: opts[:equal_nan]) ==
             Nx.tensor(1, type: {:u, 8}) do
      flunk("""
      expected

      #{inspect(lhs)}

      to be within tolerance of

      #{inspect(rhs)}
      """)
    end
  end

  def assert_lists_equal(actual_list, expected_list, delta \\ 0.001) do
    assert length(actual_list) == length(expected_list)

    Enum.zip(actual_list, expected_list)
    |> Enum.map(fn {actual, expected} ->
      assert_in_delta(actual, expected, delta)
    end)
  end

  def assert_nx_lists_equal(actual_list, expected_list, opts \\ []) do
    assert length(actual_list) == length(expected_list)

    Enum.zip(actual_list, expected_list)
    |> Enum.map(fn {actual, expected} ->
      assert_all_close(actual, expected, opts)
    end)
  end

  def read_csv(filename) do
    File.stream!(filename)
    |> CSV.decode!(field_transform: &String.trim/1)
    |> Enum.to_list()
    |> List.flatten()
    |> Enum.map(&String.to_float(String.trim(&1)))
  end

  def read_nx_list(filename) do
    File.stream!(filename)
    |> CSV.decode!(field_transform: &String.trim/1)
    |> Enum.to_list()
    |> Enum.map(fn row ->
      row |> Enum.map(&String.to_float(&1)) |> Nx.tensor()
    end)
  end

  def write_t(result, filename) do
    data = result.output_t |> Enum.join("\n")
    File.write!(filename, data)
  end

  def write_x(result, filename) do
    [first_x | _rest_of_x] = result.output_x
    {length_of_x} = Nx.shape(first_x)

    data =
      result.output_x
      |> Enum.map(fn x ->
        0..(length_of_x - 1)
        |> Enum.reduce("", fn i, acc ->
          acc <> "#{Nx.to_number(x[i])}  "
        end)
      end)
      |> Enum.join("\n")

    File.write!(filename, data)
  end
end
