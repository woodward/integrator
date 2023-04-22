defmodule Integrator.RungeKutta.BogackiShampine23Test do
  @moduledoc false
  use Integrator.TestCase

  import Nx, only: :sigils
  alias Integrator.Utils
  alias Integrator.RungeKutta.BogackiShampine23

  test "order/0" do
    assert BogackiShampine23.order() == 3
  end

  describe "integrate" do
    test "gives the correct values" do
      t = Nx.tensor(19.72183417709078, type: :f64)
      x = Nx.tensor([1.923461755449107, 0.747222633689152], type: :f64)
      dt = Nx.tensor(8.681595746273718e-02, type: :f64)

      k_vals = ~M[
         1.079890741687383   0.903592427759665   0.829725464741113   0.747222633689152
        -4.474844799526085  -4.233168820979167  -4.087042667720291  -3.940742528893131
      ]f64

      {t_next, x_next, x_est, k} = BogackiShampine23.integrate(&van_der_pol_fn/2, t, x, dt, k_vals)

      expected_t_next = Nx.tensor(19.80865013455352, type: :f64)
      expected_x_next = ~V[ 1.974378491284494   0.435401764107805 ]f64
      expected_x_est = ~V[ 1.974483141002339   0.435472329211931 ]f64

      expected_k = ~M[
         0.747222633689152   0.576162965809159   0.513870123583258   0.435401764107805
        -3.940742528893131  -3.583865100776404  -3.423158452173967  -3.236247007818676
      ]f64

      assert_all_close(t_next, expected_t_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(k, expected_k, atol: 1.0e-15, rtol: 1.0e-15)
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

    test "the interpolate function delegates", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = Utils.hermite_cubic_interpolation(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-13, rtol: 1.0e-13)
    end
  end
end
