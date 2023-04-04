defmodule IntegratorTest do
  @moduledoc false
  use Integrator.TestCase

  describe "getting a column" do
    test "can get a column from a matrix" do
      full_matrix = Nx.iota({3, 4}) |> IO.inspect(label: "full_matrix")
      column = Nx.slice_along_axis(full_matrix, 0, 1, axis: 1)
      IO.inspect(column, label: "column")
      column_reshaped = Nx.flatten(column)
      IO.inspect(column_reshaped, label: "column_reshaped")
    end
  end
end
