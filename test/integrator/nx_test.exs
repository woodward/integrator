defmodule Integrator.NxTest do
  @moduledoc false

  use Integrator.TestCase, async: true

  import Nx.Defn
  import Nx, only: :sigils

  describe "getting a column" do
    test "can get a column from a matrix" do
      # full_matrix = Nx.iota({3, 4})
      # IO.inspect(full_matrix, label: "full_matrix")
      # first_column = Nx.slice_along_axis(full_matrix, 0, 1, axis: 1)
      # IO.inspect(first_column, label: "first_column")
      # column_reshaped = Nx.flatten(first_column)
      # IO.inspect(column_reshaped, label: "column_reshaped")

      # second_column = Nx.slice_along_axis(full_matrix, 1, 1, axis: 1)
      # IO.inspect(second_column, label: "second_column")
      # second_column_reshaped = Nx.flatten(second_column)
      # IO.inspect(second_column_reshaped, label: "second_column_reshaped")

      # last_column = Nx.slice_along_axis(full_matrix, 3, 1, axis: 1)
      # IO.inspect(last_column, label: "last_column")
      # last_column_reshaped = Nx.flatten(last_column)
      # IO.inspect(last_column_reshaped, label: "last_column_reshaped")
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

    test "does stack work " do
      t1 = Nx.tensor(0.1)
      t2 = Nx.tensor(0.2)
      _t_stack = Nx.stack([t1, t2])
      # IO.inspect(t_stack)

      x1 = Nx.tensor([1, 2, 3])
      x2 = Nx.tensor([4, 5, 6])
      _x_stack = Nx.stack([x1, x2]) |> Nx.transpose()
      # IO.inspect(x_stack)
    end
  end

  describe "stack along different axis" do
    setup do
      k1 = Nx.tensor([1, 2])
      k2 = Nx.tensor([3, 4])
      k3 = Nx.tensor([5, 6])
      %{k1: k1, k2: k2, k3: k3}
    end

    test "works - old way", %{k1: k1, k2: k2, k3: k3} do
      old_way = stack_old_way(k1, k2, k3)
      expected = ~MAT[
        1 3 5
        2 4 6 ]s64
      assert_all_close(old_way, expected, atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "works - new way", %{k1: k1, k2: k2, k3: k3} do
      new_way = stack_new_way(k1, k2, k3)
      # IO.inspect(new_way)
      expected = ~MAT[
        1 3 5
        2 4 6 ]s64
      assert_all_close(new_way, expected, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  defn stack_new_way(k1, k2, k3) do
    Nx.stack([k1, k2, k3], axis: 1)
  end

  defn stack_old_way(k1, k2, k3) do
    Nx.stack([k1, k2, k3]) |> Nx.transpose()
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

  describe "slice to get part of a vector" do
    test "works" do
      vec = Nx.tensor([1, 2, 3, 4, 5])
      _sliced = Nx.slice_along_axis(vec, 1, 4, axis: 0)
      # IO.inspect(sliced)
    end
  end

  describe "Nx.dot/4" do
    test "works better than dot, transpose" do
      k_0_4 = ~MAT[
        0.09819789615529644 0.09728161794756425 0.09682469517700483 0.09454141156537164 0.09413618004462349
       -2.303556017851066  -2.3008378798836384 -2.299481814584658  -2.2927038466248972 -2.2915004503322134  ]f64

      aa_5 =
        ~VEC[ 0.00566076962506267  -0.021395034803386472  0.017713398289163906  5.53709527482009e-4  -5.440084171622433e-4]f64

      expected = ~VEC[ 1.907640620018197e-4 -0.004567927089507702]f64

      old_way = Nx.dot(k_0_4, Nx.transpose(aa_5))
      assert_all_close(old_way, expected, atol: 1.0e-15, rtol: 1.0e-15)

      # Examples from docs:
      # t1 = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      # t1 = [1, 2]
      #      [3, 4]
      # t2 = Nx.tensor([[10, 20], [30, 40]], names: [:height, :width])
      # t2 = [10, 20]
      #      [30, 40]
      # result = Nx.dot(t1, [0], t2, [0])
      # IO.inspect(result)

      # t1 = Nx.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
      # t1 = [0.0, 1.0, 2.0]
      #      [3.0, 4.0, 5.0]
      #
      # t2 = Nx.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
      # t2 = [0.0, 1.0]
      #      [2.0, 3.0]
      #      [4.0, 5.0]
      # result2 = Nx.dot(t1, [0, 1], t2, [1, 0])
      # IO.inspect(result2)
      # result2 = 50

      new_way = Nx.dot(k_0_4, [1], aa_5, [0])
      assert_all_close(new_way, expected, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end
end
