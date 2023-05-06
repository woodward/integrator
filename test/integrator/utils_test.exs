defmodule Integrator.UtilsTest do
  @moduledoc false
  use Integrator.TestCase
  import Nx, only: :sigils

  alias Integrator.Utils

  describe "hermite_quartic_interpolation" do
    setup do
      # These test values were obtained from Octave:
      # Generated using:
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset("AbsTol", 1.0e-14, "RelTol", 1.0e-14)
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0], opts);

      t = ~V[ 19.97226029930081   19.97424839002798 ]f64

      x = ~M[ 2.008585111348593e+00   2.008604708104012e+00
              1.188547490189183e-02   7.832739209072674e-03
            ]f64

      der = ~M[
        1.188547490189183e-02   1.107248473635211e-02   1.066709096215445e-02   8.641324907205110e-03   8.281808253394873e-03     7.832711009917654e-03   7.832739209072674e-03
       -2.044650564564792e+00  -2.042188551791212e+00  -2.040960496435665e+00  -2.034823361858414e+00  -2.033733968850626e+00    -2.032373024665618e+00  -2.032373099413282e+00
      ]f64

      t_out = ~V[ 19.97275732198261   19.97325434466440   19.97375136734619   19.97424839002798 ]f64

      expected_x_out = ~M[
       2.008590766279272e+00   2.008595916876415e+00   2.008600563898753e+00   2.008604708104012e+00
       1.087000165112446e-02   9.856055965852775e-03   8.843635825513105e-03   7.832739209072674e-03
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Utils.hermite_quartic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-13, rtol: 1.0e-13)
    end

    test "works for a single value of t (rather than an array of t)", %{t: t, x: x, der: der} do
      t_out = ~V[ 19.97375136734619 ]f64

      expected_x_out = ~M[
        2.008600563898753e+00
        8.843635825513105e-03
      ]f64

      x_out = Utils.hermite_quartic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-14, rtol: 1.0e-14)
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
