defmodule Integrator.RungeKutta.DormandPrince45Test do
  @moduledoc false
  use Integrator.TestCase

  import Nx, only: :sigils
  alias Integrator.RungeKutta.DormandPrince45

  test "order/0" do
    assert DormandPrince45.order() == 5
  end

  describe "integrate/5" do
    test "gives the correct result for the van der pol function" do
      t = Nx.tensor(19.711, type: :f64)
      x = Nx.tensor([1.9265, 0.7353], type: :f64)
      dt = Nx.tensor(0.2893, type: :f64)

      k_vals = ~M[
         2.1068   1.8570   1.6938   1.0933   1.2077   1.0641   0.7353
        -4.1080  -4.6688  -4.7104  -4.4778  -4.5190  -4.3026  -3.9202
      ]f64

      {t_next, x_next, x_est, k} = DormandPrince45.integrate(&van_der_pol_fn/2, t, x, dt, k_vals)

      expected_t_next = Nx.tensor(20.0, type: :f64)
      expected_x_next = ~V[ 2.007378 -0.071766 ]f64
      expected_x_est = ~V[ 2.007393 -0.071892 ]f64

      expected_k = ~M[
           0.735294   0.508467   0.426833   0.026514  -0.057984  -0.116136  -0.071766
          -3.920232  -3.432015  -3.214602  -2.106530  -1.871609  -1.681243  -1.789958
      ]f64

      assert_all_close(t_next, expected_t_next, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(k, expected_k, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "gives the correct result when there are no existing k_vals" do
      # Used Octave function:
      #  [t,x] = ode45 (fvdp, [0, 20], [2, 1]);
      # i.e., the initial values for x have been changed from [2, 0] to [2, 1]
      # (so that both x values are non-zero)

      t = Nx.tensor(0.0, type: :f64)
      x = Nx.tensor([2.0, 1.0], type: :f64)
      dt = Nx.tensor(0.068129, type: :f64)

      k_vals = ~M[
         0.0 0.0 0.0 0.0 0.0 0.0 0.0
         0.0 0.0 0.0 0.0 0.0 0.0 0.0
       ]f64

      {t_next, x_next, x_est, k} = DormandPrince45.integrate(&van_der_pol_fn/2, t, x, dt, k_vals)

      expected_t_next = Nx.tensor(0.068129, type: :f64)
      expected_x_next = ~V[ 2.0571  0.6839 ]f64
      expected_x_est = ~V[ 2.0571   0.6839 ]f64

      expected_k = ~M[
           1.0000   0.9319   0.8999   0.7429   0.7162   0.6836   0.6839
          -5.0000  -4.8602  -4.7894  -4.4195  -4.3536  -4.2690  -4.2670
        ]f64

      assert_all_close(t_next, expected_t_next, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(k, expected_k, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "gives the correct result for the euler_equations" do
      t = Nx.tensor(3.162277660168380e-02, type: :f64)
      x = Nx.tensor([3.161482041589618e-02, 9.995001266279027e-01, 9.997450958099772e-01], type: :f64)
      dt = Nx.tensor(4.743416490252569e-02, type: :f64)

      k_vals = ~M[
        1.000000000000000   1.000000000000000   0.999932051032750   0.999516901172172   0.999403489664703    0.999245108800132   0.999245349857697
        0.0                -0.006324555320337  -0.009486615257688  -0.025286454530879  -0.028086810350429   -0.031596486671431  -0.031606761665705
        0                  -0.003225523213372  -0.004838067097241  -0.012894069373540  -0.014321499330348   -0.016110258607291  -0.016115498674593
      ]f64

      {t_next, x_next, x_est, k} = DormandPrince45.integrate(&euler_equations/2, t, x, dt, k_vals)

      expected_t_next = Nx.tensor(7.905694150420950e-02, type: :f64)
      expected_x_next = ~V[ 7.893280729541992e-02  9.968799385689663e-01  9.984099869704532e-01 ]f64
      expected_x_est = ~V[  7.893280780208246e-02  9.968799385539475e-01  9.984099869640367e-01]f64

      expected_k = ~M[
         9.992453498576969e-01   9.987928154423733e-01   9.984140992639023e-01   9.963546001047874e-01    9.959014391234161e-01   9.952960644632111e-01   9.952948864777479e-01
        -3.160676166570535e-02  -4.107773626423464e-02  -4.580494868962854e-02  -6.938788802174620e-02   -7.354618661114984e-02  -7.877273023115196e-02  -7.880730310336150e-02
        -1.611549867459338e-02  -2.094143130431229e-02  -2.334848697501391e-02  -3.534583814940384e-02   -3.745849936749036e-02  -4.011253394624428e-02  -4.013013136474447e-02
      ]f64

      assert_all_close(t_next, expected_t_next, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-04, rtol: 1.0e-04)
      assert_all_close(k, expected_k, atol: 1.0e-04, rtol: 1.0e-04)
    end
  end

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

    test "the interpolate function delegates", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = DormandPrince45.interpolate(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-9, rtol: 1.0e-9)
    end
  end
end