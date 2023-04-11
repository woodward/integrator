defmodule Integrator.AdaptiveStepsizeTest do
  @moduledoc false
  use Integrator.TestCase

  alias Integrator.{AdaptiveStepsize, Test}
  alias Integrator.RungeKutta.DormandPrince45

  describe "integrate" do
    test "works" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Test.van_der_pol_fn/2

      t_start = 0.0
      t_end = 2.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = []
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      temp = result.temp
      assert_all_close(temp.dt, 0.068129)
      assert_all_close(temp.factor, Nx.tensor(0.8511))

      # assert result.count_loop == 50
      # assert result.count_cyles == 78
      # assert result.count_save == 2
      # assert result.unhandled_termination == 1

      # expected_t = read_csv("test/fixtures/integrator/integrator/runge_kutta_45_test/time.csv")
      # expected_y = read_nx_list("test/fixtures/integrator/integrator/runge_kutta_45_test/y.csv")

      # assert_lists_equal(result.output_t, expected_t)
      # assert_all_close(result.output_y, expected_y)
    end
  end

  describe "kahan_sum" do
    test "sums up some items" do
      sum = Nx.tensor(2.74295650014, type: :f64)
      comp = Nx.tensor(1.11022302463e-16, type: :f64)
      term = Nx.tensor(0.66059601818, type: :f64)

      expected_sum = Nx.tensor(3.40355251832, type: :f64)
      expected_comp = Nx.tensor(1.11022302463e-16, type: :f64)

      {sum, comp} = AdaptiveStepsize.kahan_sum(sum, comp, term)

      assert_all_close(sum, expected_sum, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(comp, expected_comp, atol: 1.0e-14, rtol: 1.0e-14)
    end
  end
end
