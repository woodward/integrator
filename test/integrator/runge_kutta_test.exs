defmodule Integrator.RungeKuttaTest do
  @moduledoc false
  use Integrator.TestCase

  import Nx, only: :sigils
  alias Integrator.RungeKutta

  describe "interpolate/7" do
    test "works for Dormand-Prince45" do
      # Note that these values were captured during a refactor via dbg statements, and then were turned
      # into a test case for regression purposes.  If possible, a test case based on values from
      # another source (e.g., Octave) should be created.

      t_add = ~VEC[3.1933543442858077  3.488060218650614  3.78276609301542  4.077471967380226]f64
      interpolate_fn = &RungeKutta.DormandPrince45.interpolate/4

      t_old = Nx.f64(2.898648469921002)
      x_old = ~VEC[16.76036011799986  -8.435741489925032]f64

      t_new = Nx.f64(4.2941803179443205)
      x_new = ~VEC[-4.564518118928589  -22.125908919033794]f64

      k_vals = ~MAT[
          -8.435741489925032  -11.173774975746785  -12.54279171865766  -19.387875433212034  -20.604779204688395  -22.12590891903377  -22.125908919033794
          -9.81                -9.81                -9.81               -9.81                -9.81                -9.81               -9.81
      ]f64

      expected_result = ~MAT[
          13.848290681846994   10.084207516796049   5.468110622847037   -3.6415315207705135e-14
         -11.326806117443777  -14.217870744962527 -17.108935372481277  -20.00000000000002
      ]f64

      result = RungeKutta.interpolate(t_add, interpolate_fn, t_old, x_old, t_new, x_new, k_vals)
      assert_all_close(result, expected_result, atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "works for Bogacki-Shampine23" do
      # Note that these values were captured during a refactor via dbg statements, and then were turned
      # into a test case for regression purposes.  If possible, a test case based on values from
      # another source (e.g., Octave) should be created.

      t_add = ~VEC[19.945287340231687  19.958965505173765  19.972643670115843  19.98632183505792]f64
      interpolate_fn = &RungeKutta.BogackiShampine23.interpolate/4

      t_old = Nx.f64(19.93160917528961)
      x_old = ~VEC[2.006095079498686  0.0965171259321903]f64

      t_new = Nx.f64(19.98632183505792)
      x_new = ~VEC[2.0081230062138307  -0.019464541184915612]f64

      k_vals = ~MAT[
            0.0965171259321903 0.033652193381535855 0.009898623457363892  -0.019464541184915612
           -2.298003161128081 -2.1108704479395213  -2.0374684115168598    -1.9490956559698092
      ]f64

      expected_result = ~MAT[
          2.0072044625559693   2.0079022510428013    2.008203435436361       2.0081230062138307
          0.06571298789094157  0.036140085171095404  0.0077602860525202155  -0.019464541184915612
      ]f64

      result = RungeKutta.interpolate(t_add, interpolate_fn, t_old, x_old, t_new, x_new, k_vals)
      assert_all_close(result, expected_result, atol: 1.0e-14, rtol: 1.0e-14)
    end
  end
end
