defmodule Integrator.NxTest do
  @moduledoc false
  use Integrator.TestCase

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
end
