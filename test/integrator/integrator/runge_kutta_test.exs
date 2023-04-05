defmodule Integrator.RungeKuttaTest do
  @moduledoc false
  use Integrator.TestCase

  import Nx, only: :sigils
  alias Integrator.RungeKutta

  describe "dormand_prince" do
    test "gives the correct result" do
      t = Nx.tensor(19.711, type: :f64)
      x = Nx.tensor([1.9265, 0.7353], type: :f64)
      dt = Nx.tensor(0.2893, type: :f64)

      k_vals = ~M[
         2.1068   1.8570   1.6938   1.0933   1.2077   1.0641   0.7353
        -4.1080  -4.6688  -4.7104  -4.4778  -4.5190  -4.3026  -3.9202
      ]f64

      {t_next, x_next, x_est, k} = RungeKutta.dormand_prince_45(&van_der_pol_fn/2, t, x, dt, k_vals)

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

    test "works when evaluating the first timestep" do
    end
  end

  describe "hermite_quartic_interpolation" do
    test "gives the correct result" do
      # Values from Octave:
      #
      # t = [ 19.4067624192,   19.7106968201 ]
      #
      #  x =
      #
      #    1.49652504841   1.92651431954
      #    2.10676183153   0.73529371547
      #
      #  der =
      #
      #     2.10676183153   1.85704689071   1.69384014677   1.09328301986   1.20767740745   1.06405291882   0.73529371547
      #    -4.10804009148  -4.66882299445  -4.71039008294  -4.47781878341  -4.51898062008  -4.30261858646  -3.92023192271
      #
      #  t_out = [ 19.4827460194,   19.5587296196, 19.6347132198,  19.7106968201 ]
      #
      #  x_out =
      #
      #   1.64398703647   1.76488148020   1.85862355568   1.92651431954
      #   1.77097584066   1.41075566029   1.05684789008   0.73529371547

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

      x_out = RungeKutta.hermite_quartic_interpolation(t, x, der, t_out)

      assert_all_close(x_out, expected_x_out, atol: 1.0e-9, rtol: 1.0e-9)
    end
  end
end
