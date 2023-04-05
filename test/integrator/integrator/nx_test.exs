defmodule Integrator.NxTest do
  @moduledoc false
  use Integrator.TestCase
  import Nx.Defn

  describe "getting a column" do
    test "can get a column from a matrix" do
      full_matrix = Nx.iota({3, 4})
      # IO.inspect(full_matrix, label: "full_matrix")
      first_column = Nx.slice_along_axis(full_matrix, 0, 1, axis: 1)
      # IO.inspect(first_column, label: "first_column")
      _column_reshaped = Nx.flatten(first_column)
      # IO.inspect(column_reshaped, label: "column_reshaped")

      _second_column = Nx.slice_along_axis(full_matrix, 1, 1, axis: 1)
      # IO.inspect(second_column, label: "second_column")
    end
  end

  describe "dot" do
    test "dots" do
      full_matrix = Nx.iota({3, 4})
      # IO.inspect(full_matrix)
      column = Nx.tensor([1, 2, 3, 4])
      _dot = Nx.dot(full_matrix, column)
      # IO.inspect(dot)
    end
  end

  describe "turn row into column" do
    test "works" do
      full_matrix = Nx.tensor([1, 2, 3])
      # IO.inspect(full_matrix, label: "full_matrix")

      _column = Nx.new_axis(full_matrix, 1)
      # IO.inspect(column)
    end
  end

  describe "create a zero matrix with a certain number of rows" do
    test "create it" do
      x = Nx.tensor([1, 2])
      {length_of_x} = Nx.shape(x)
      _with_zeros = Nx.broadcast(0.0, {length_of_x, 7})

      # IO.inspect(with_zeros, label: "with_zeros")
    end
  end

  describe "does stack work in defn?" do
    test "check on it" do
      k1 = Nx.tensor([1, 2])
      k2 = Nx.tensor([3, 4])
      _k = try_to_stack(k1, k2)
      # IO.inspect(k)
    end
  end

  defn try_to_stack(k1, k2) do
    Nx.stack([k1, k2]) |> Nx.transpose()
  end

  describe "get an x-y element of a tensor" do
    test "works" do
      full_matrix = Nx.iota({3, 4})
      # IO.inspect(full_matrix, label: "full_matrix")

      _x_2_3 = full_matrix[1][2]

      # IO.inspect(x_2_3, label: "x_2_3")

      _x_2_3 = x_2_3_in_defn(full_matrix)
      # IO.inspect(x_2_3, label: "x_2_3")
    end
  end

  def x_2_3_in_defn(x) do
    Nx.stack([x[1][2], x[1][3]])
  end

  describe "getting a sum of a column" do
    test "it works" do
      vec = Nx.tensor([1.0, 2.0, -3.0])
      _sum = Nx.abs(vec) |> Nx.sum()
      # IO.inspect(sum)
    end
  end

  describe "slicing to get specific rows" do
    test "works" do
      full_matrix = Nx.iota({6, 6})
      # IO.inspect(full_matrix, label: "full_matrix")
      _most_of_last_row = Nx.slice_along_axis(full_matrix, 5, 1) |> Nx.flatten() |> Nx.slice_along_axis(0, 5)
      # IO.inspect(most_of_last_row, label: "most_of_last_row")

      _next_to_last_row = Nx.slice_along_axis(full_matrix, 4, 1) |> Nx.flatten() |> Nx.slice_along_axis(0, 4)
      # IO.inspect(next_to_last_row, label: "next_to_last_row")
    end
  end
end
