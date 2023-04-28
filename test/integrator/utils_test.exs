defmodule Integrator.UtilsTest do
  @moduledoc false
  use Integrator.TestCase
  import Nx, only: :sigils

  alias Integrator.Utils

  describe "hermite_quartic_interpolation" do
    setup do
      # These test values were obtained from Octave:

      t = ~V[ 19.4067624192  19.7106968201 ]f64

      x = ~M[ 1.49652504841   1.92651431954
              2.10676183153   0.73529371547
            ]f64

      der = ~M[
           2.10676183153   1.85704689071   1.69384014677   1.09328301986   1.20767740745   1.06405291882   0.73529371547
          -4.10804009148  -4.66882299445  -4.71039008294  -4.47781878341  -4.51898062008  -4.30261858646  -3.92023192271
      ]f64

      t_out = ~V[ 19.4827460194  19.5587296196  19.6347132198  19.7106968201 ]f64

      expected_x_out = ~M[
        1.64398703647   1.76488148020   1.85862355568   1.92651431954
        1.77097584066   1.41075566029   1.05684789008   0.73529371547
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Utils.hermite_quartic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-9, rtol: 1.0e-9)
    end
  end

  describe "hermite_cubic_interpolation" do
    setup do
      # These test values were obtained from Octave:

      t = ~V[ 19.93160917528972   19.98632183505794 ]f64

      x = ~M[ 2.006095079498698e+00   2.008123006213831e+00
              9.651712593193962e-02  -1.946454118495647e-02
          ]f64

      der = ~M[
        9.651712593193962e-02   3.365219338140671e-02   9.898623457268760e-03  -1.946454118495647e-02
       -2.298003161127339e+00  -2.110870447939130e+00  -2.037468411516576e+00  -1.949095655969686e+00
      ]f64

      t_out = ~V[ 19.94528734023178   19.95896550517383   19.97264367011589   19.98632183505794 ]f64

      expected_x_out = ~M[
        2.007204462555976e+00   2.007902251042804e+00   2.008203435436362e+00   2.008123006213831e+00
        6.571298789074770e-02   3.614008517096150e-02   7.760286052434550e-03  -1.946454118495647e-02
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Utils.hermite_cubic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-13, rtol: 1.0e-13)
    end
  end

  describe "kahan_sum" do
    test "sums up some items" do
      sum = Nx.tensor(2.74295650014, type: :f64)
      comp = Nx.tensor(1.11022302463e-16, type: :f64)
      term = Nx.tensor(0.66059601818, type: :f64)

      expected_sum = Nx.tensor(3.40355251832, type: :f64)
      expected_comp = Nx.tensor(1.11022302463e-16, type: :f64)

      {sum, comp} = Utils.kahan_sum(sum, comp, term)

      assert_all_close(sum, expected_sum, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(comp, expected_comp, atol: 1.0e-14, rtol: 1.0e-14)
    end
  end

  describe "columns_as_list" do
    test "works" do
      matrix = Nx.iota({2, 5})
      cols_as_list = Utils.columns_as_list(matrix, 1, 3)

      expected_cols_as_list = [
        # Nx.tensor([0, 5]),
        Nx.tensor([1, 6]),
        Nx.tensor([2, 7]),
        Nx.tensor([3, 8])
        # Nx.tensor([4, 9]),
      ]

      assert cols_as_list == expected_cols_as_list
    end

    test "goes all the way to the end if the end_index is left out" do
      matrix = Nx.iota({2, 5})
      cols_as_list = Utils.columns_as_list(matrix, 1)

      expected_cols_as_list = [
        # Not present: Nx.tensor([0, 5]),
        Nx.tensor([1, 6]),
        Nx.tensor([2, 7]),
        Nx.tensor([3, 8]),
        Nx.tensor([4, 9])
      ]

      assert cols_as_list == expected_cols_as_list
    end
  end

  describe "vector_as_list" do
    test "works" do
      vector = Nx.tensor([1, 2, 3])
      vector_as_list = vector |> Utils.vector_as_list()

      assert vector_as_list == [
               Nx.tensor(1),
               Nx.tensor(2),
               Nx.tensor(3)
             ]
    end
  end

  describe "sign" do
    test "is negative one for things less than one" do
      assert Utils.sign(-7.0) == -1.0
    end

    test "is plus one for things less than one" do
      assert Utils.sign(7.0) == 1.0
    end

    test "is zero for zero" do
      assert Utils.sign(0.0) == 0.0
    end
  end

  describe "unique/1" do
    test "returns all values if they are unique" do
      assert Utils.unique([3.3, 2.2, 1.1]) == [1.1, 2.2, 3.3]
    end

    test "only returns unique values sorted in ascending order" do
      assert Utils.unique([3.3, 2.2, 3.3]) == [2.2, 3.3]
    end
  end

  describe "type_atom" do
    # Maybe there's a built-in Nx way of doing this (seems like there should be) but I can't find it

    test "returns the Nx type as an atom for :f32 tensors" do
      nx_f32 = Nx.tensor(0.1, type: :f32)
      assert Utils.type_atom(nx_f32) == :f32
    end

    test "returns the Nx type as an atom for :f64 tensors" do
      nx_f64 = Nx.tensor(0.1, type: :f64)
      assert Utils.type_atom(nx_f64) == :f64
    end
  end
end
