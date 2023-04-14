defmodule Integrator.AdaptiveStepsizeTest do
  @moduledoc false
  use Integrator.TestCase
  import Nx, only: :sigils

  alias Integrator.{AdaptiveStepsize, Test}
  alias Integrator.RungeKutta.DormandPrince45

  describe "integrate" do
    test "works" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Test.van_der_pol_fn/2

      t_start = 0.0
      # t_end = 4.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = []
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert result.count_save == 2
      assert result.unhandled_termination == true

      expected_t = read_csv("test/fixtures/integrator/integrator/runge_kutta_45_test/time.csv")
      expected_x = read_nx_list("test/fixtures/integrator/integrator/runge_kutta_45_test/x.csv")

      assert_lists_equal(result.output_t, expected_t, 0.01)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 0.1, rtol: 0.1)
    end

    test "works - high fidelity" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Test.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [abs_tol: 1.0e-10, rel_tol: 1.0e-10]
      initial_tstep = 0.007418363820761442

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 1037
      assert result.count_loop__increment_step == 1027
      assert result.count_save == 2
      assert result.unhandled_termination == true

      expected_t = read_csv("test/fixtures/integrator/integrator/runge_kutta_45_test/time_high_fidelity.csv")
      expected_x = read_nx_list("test/fixtures/integrator/integrator/runge_kutta_45_test/x_high_fidelity.csv")

      # data = result.output_t |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/time_high_fidelity-elixir.csv", data)
      #
      # data = result.output_x |> Enum.map(fn xn -> "#{Nx.to_number(xn[0])}  #{Nx.to_number(xn[1])}  " end) |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/x_high_fidelity-elixir.csv", data)
      assert_lists_equal(result.output_t, expected_t, 1.0e-02)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-02, rtol: 1.0e-02)
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

  describe "compute_next_timestep" do
    test "basic case" do
      dt = 0.068129
      error = 0.0015164936598390992
      order = 5
      t_old = 0.0
      t_end = 2.0

      new_dt = AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, epsilon: 2.2204e-16)

      expected_dt = 0.1022
      assert_in_delta(new_dt, expected_dt, 1.0e-05)
    end

    test "uses option :max_step" do
      dt = 0.068129
      error = 0.0015164936598390992
      order = 5
      t_old = 0.0
      t_end = 2.0

      new_dt = AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, max_step: 0.05, epsilon: 2.2204e-16)

      expected_dt = 0.05
      assert_in_delta(new_dt, expected_dt, 1.0e-05)
    end

    test "does not go past t_end" do
      dt = 0.3039
      error = 0.4414
      order = 5
      t_old = 19.711
      t_end = 20.0

      # fac = 0.8511
      # dt_after_times_min = 0.2964
      # options_maxstep = 2
      # tspan_end = 20
      # t_old = 19.711
      # abs_tspan_end = 0.2893
      # dt = 0.2893

      new_dt = AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, epsilon: 2.2204e-16)

      expected_dt = 0.289
      assert_in_delta(new_dt, expected_dt, 1.0e-05)
    end
  end

  describe "initial_empty_k_vals" do
    test "returns a tensor with zeros that's the correct size" do
      order = 5
      x = ~V[ 1.0 2.0 3.0 ]f64
      k_vals = AdaptiveStepsize.initial_empty_k_vals(order, x)

      expected_k_vals = ~M[
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
      ]f64

      assert_all_close(k_vals, expected_k_vals)
    end
  end
end
