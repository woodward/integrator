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

      t = ~VEC[ 19.97226029930081   19.97424839002798 ]f64

      x = ~MAT[ 2.008585111348593e+00   2.008604708104012e+00
                1.188547490189183e-02   7.832739209072674e-03
            ]f64

      der = ~MAT[
        1.188547490189183e-02   1.107248473635211e-02   1.066709096215445e-02   8.641324907205110e-03   8.281808253394873e-03     7.832711009917654e-03   7.832739209072674e-03
       -2.044650564564792e+00  -2.042188551791212e+00  -2.040960496435665e+00  -2.034823361858414e+00  -2.033733968850626e+00    -2.032373024665618e+00  -2.032373099413282e+00
      ]f64

      t_out = ~VEC[ 19.97275732198261   19.97325434466440   19.97375136734619   19.97424839002798 ]f64

      expected_x_out = ~MAT[
       2.008590766279272e+00   2.008595916876415e+00   2.008600563898753e+00   2.008604708104012e+00
       1.087000165112446e-02   9.856055965852775e-03   8.843635825513105e-03   7.832739209072674e-03
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Utils.hermite_quartic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-13, rtol: 1.0e-13)
      assert_nx_f64(x_out)
    end

    test "works for a single value of t (rather than an array of t)", %{t: t, x: x, der: der} do
      t_out = ~VEC[ 19.97375136734619 ]f64

      expected_x_out = ~MAT[
        2.008600563898753e+00
        8.843635825513105e-03
      ]f64

      x_out = Utils.hermite_quartic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-14, rtol: 1.0e-14)
      assert_nx_f64(x_out)
    end
  end

  describe "hermite_cubic_interpolation" do
    setup do
      # These test values were obtained from Octave:
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset("AbsTol", 1.0e-14, "RelTol", 1.0e-14, "Refine", 4)
      # [t,x] = ode23 (fvdp, [0, 20], [2, 0], opts);

      t = ~VEC[ 1.393975803797305   1.394019009661092 ]f64

      x = ~MAT[    1.155084908374247   1.155040344468921
                -1.031414373710718  -1.031449387283638
          ]f64

      der = ~MAT[
         -1.031414373710718  -1.031431879957980  -1.031440633890396  -1.031449387283638
         -0.810364414851849  -0.810389373959629  -0.810401854319440  -0.810414334531038
      ]f64

      t_out = ~VEC[ 1.393986605263252   1.393997406729198   1.394008208195145   1.394019009661092 ]f64

      expected_x_out = ~MAT[
          1.155073767539739   1.155062626610683   1.155051485587077   1.155040344468921
         -1.031423126901747  -1.031431880227575  -1.031440633688204  -1.031449387283638
      ]f64

      [t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out]
    end

    test "gives the correct result", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Utils.hermite_cubic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-15, rtol: 1.0e-15)
      assert_nx_f64(x_out)
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
      assert_nx_f64(sum)
      assert_nx_f64(comp)
    end

    test "another test case" do
      # All values are taken from Octave:
      t_old = Nx.tensor(3.636484156979396e-02, type: :f64)
      options_comp_old = Nx.tensor(3.469446951953614e-18, type: :f64)
      dt = Nx.tensor(8.037014854361582e-03, type: :f64)

      {t_new, options_comp_new} = Utils.kahan_sum(t_old, options_comp_old, dt)

      expected_t_new = Nx.tensor(4.440185642415553e-02, type: :f64)
      #                          4.4401856424155534e-02  From Elixir
      # IO.inspect(Nx.to_number(t_new), label: "t_new")

      expected_options_comp_new = Nx.tensor(-1.734723475976807e-18, type: :f64)
      #                                     -1.734723475976807e-18  From Elixir
      # IO.inspect(Nx.to_number(options_comp_new), label: "options_comp_new")

      assert_all_close(t_new, expected_t_new, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(options_comp_new, expected_options_comp_new, atol: 1.0e-23, rtol: 1.0e-23)
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
      vector = Nx.tensor([1, 2, 3], type: :f64)
      vector_as_list = vector |> Utils.vector_as_list()

      assert vector_as_list == [
               Nx.tensor(1, type: :f64),
               Nx.tensor(2, type: :f64),
               Nx.tensor(3, type: :f64)
             ]

      [first | [second | [third]]] = vector_as_list
      assert_nx_f64(first)
      assert_nx_f64(second)
      assert_nx_f64(third)
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
end
