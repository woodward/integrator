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
      # Octave:
      # format long
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset("AbsTol", 1.0e-14, "RelTol", 1.0e-14)
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0], opts);

      t = ~V[  19.93252120192793  ]f64
      x = ~V[  2.006431879061176e+00   9.819789615529644e-02  ]f64
      dt = ~V[  1.988834221159869e-03  ]f64

      k_vals = ~M[
         1.027961290443385e-01   1.018737675533550e-01   1.014138104312208e-01   9.911536312201520e-02   9.870744000286301e-02    9.819786643607928e-02   9.819789615529644e-02
        -2.317186695289708e+00  -2.314454227194995e+00  -2.313091000296263e+00  -2.306277176899946e+00  -2.305067401715800e+00   -2.303555944193641e+00  -2.303556017851066e+00
      ]f64

      {t_next, x_next, x_est, k} = DormandPrince45.integrate(&van_der_pol_fn/2, t, x, dt, k_vals)

      expected_t_next = ~V[  19.93451003614909  ]f64
      expected_x_next = ~V[  2.006622631532748e+00   9.362999867595886e-02  ]f64
      expected_x_est = ~V[   2.006622631532748e+00   9.362999867595506e-02  ]f64
      expected_k = ~M[
         9.819789615529644e-02   9.728161794756425e-02   9.682469517700483e-02   9.454141156537164e-02   9.413618004462349e-02    9.362996906578873e-02   9.362999867595886e-02
        -2.303556017851066e+00  -2.300837879883639e+00  -2.299481814584659e+00  -2.292703846624898e+00  -2.291500450332214e+00   -2.289996968358019e+00  -2.289997042028566e+00
      ]f64

      assert_all_close(t_next, expected_t_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-20, rtol: 1.0e-20)
      assert_all_close(k, expected_k, atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "gives the correct result when there are no existing k_vals" do
      # Used Octave function:
      # format long
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset("AbsTol", 1.0e-14, "RelTol", 1.0e-14)
      # [t,x] = ode45 (fvdp, [0, 20], [2, 1], opts);
      # i.e., the initial values for x have been changed from [2, 0] to [2, 1]
      # (so that both x values are non-zero)

      t = ~V[  0.0  ]f64
      x = ~V[  2.0 1.0  ]f64
      dt = ~V[  1.463190090188842e-03  ]f64

      k_vals = ~M[
         0.0 0.0 0.0 0.0 0.0 0.0 0.0
         0.0 0.0 0.0 0.0 0.0 0.0 0.0
       ]f64

      {t_next, x_next, x_est, k} = DormandPrince45.integrate(&van_der_pol_fn/2, t, x, dt, k_vals)

      expected_t_next = ~V[ 1.463190090188842e-03 ]f64
      expected_x_next = ~V[ 2.001457843004331  0.992694771307345 ]f64
      expected_x_est = ~V[  2.001457843004330  0.992694771307346 ]f64

      expected_k = ~M[
         1.000000000000000   0.998536809909811   0.997806178816787   0.994154098216414   0.993505399340018   0.992694767971552   0.992694771307345
        -5.000000000000000  -4.997071992591137  -4.995607257137799  -4.988272015168613  -4.986966166273387  -4.985333062714751  -4.985333039217887
        ]f64

      assert_all_close(t_next, expected_t_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(k, expected_k, atol: 1.0e-15, rtol: 1.0e-15)
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

      assert_all_close(t_next, expected_t_next, atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-20, rtol: 1.0e-20)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(k, expected_k, atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "gives the correct result for the ballode high fidelity example - first integration" do
      t = ~V[  0.0  ]f64
      x = ~V[  0.0  20.0  ]f64
      dt = ~V[  0.001472499532027109  ]f64
      #                    xxxxxxxxx
      #         0.001472499236077083
      #         0.001472499236077083

      k_vals = ~M[
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
      ]f64

      ode_fn = fn _t, x ->
        x0 = x[1]
        x1 = Nx.tensor(-9.81, type: :f64)
        Nx.stack([x0, x1])
      end

      {t_next, x_next, x_est, k} = DormandPrince45.integrate(ode_fn, t, x, dt, k_vals)

      # Expected values are from Octave:
      expected_t_next = ~V[  1.472499532027109e-03  ]f64
      expected_x_next = ~V[  0.02943935535039590  19.98555477959081  ]f64
      #                     [0.0294393553503959,  19.985554779590814]       from this test
      #                     [0.02943934943567045, 19.985554782494084]       from actual simulation
      expected_x_est = ~V[   0.02943935535039590  19.98555477959081  ]f64

      expected_k = ~M[
         2.000000000000000e+01   1.999711095591816e+01   1.999566643387724e+01   1.998844382367265e+01   1.998715980408072e+01   1.998555477959081e+01   1.998555477959081e+01
        -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00
      ]f64

      assert_all_close(t_next, expected_t_next, atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(x_next, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(x_est, expected_x_est, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(k, expected_k, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

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

    test "the interpolate function delegates", %{t: t, x: x, der: der, t_out: t_out, expected_x_out: expected_x_out} do
      x_out = DormandPrince45.interpolate(t, x, der, t_out)
      assert_all_close(x_out, expected_x_out, atol: 1.0e-13, rtol: 1.0e-13)
    end
  end
end
