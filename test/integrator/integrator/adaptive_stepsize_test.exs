defmodule Integrator.AdaptiveStepsizeTest do
  @moduledoc false
  use Integrator.DemoCase
  import Nx, only: :sigils

  alias Integrator.{AdaptiveStepsize, Demo, DummyOutput}
  alias Integrator.RungeKutta.DormandPrince45

  describe "integrate" do
    test "works" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = []

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert result.count_save == 2
      assert result.unhandled_termination == true
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 201
      assert length(result.output_x) == 201

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_in_delta(last_time, 20.0, 1.0e-10)

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_lists_equal(result.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - high fidelity" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [abs_tol: 1.0e-10, rel_tol: 1.0e-10]

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.007418363820761442

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 1037
      assert result.count_loop__increment_step == 1027
      assert result.count_save == 2
      assert result.unhandled_termination == true
      assert length(result.ode_t) == 1028
      assert length(result.ode_x) == 1028
      assert length(result.output_t) == 4109
      assert length(result.output_x) == 4109

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/x.csv")

      # data = result.output_t |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/time_high_fidelity-elixir.csv", data)
      #
      # data = result.output_x |> Enum.map(fn xn -> "#{Nx.to_number(xn[0])}  #{Nx.to_number(xn[1])}  " end) |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/x_high_fidelity-elixir.csv", data)
      assert_lists_equal(result.output_t, expected_t, 1.0e-05)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - no data interpolation (refine == 1)" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [refine: 1]

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert result.count_save == 2
      assert result.unhandled_termination == true
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 51
      assert length(result.output_x) == 51

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/x.csv")

      # data = result.output_t |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/time_refine_1-elixir.csv", data)

      # data = result.output_x |> Enum.map(fn xn -> "#{Nx.to_number(xn[0])}  #{Nx.to_number(xn[1])}  " end) |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/x_refine_1-elixir.csv", data)
      assert_lists_equal(result.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - output function with interpolation" do
      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, x -> DummyOutput.add_data(dummy_output_name, %{t: t, x: x}) end

      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      # t_end = 4.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [output_fn: output_fn]

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert result.count_save == 2
      assert result.unhandled_termination == true
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 201
      assert length(result.output_x) == 201

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_in_delta(last_time, 20.0, 1.0e-10)

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_lists_equal(result.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 201
      assert length(t_data) == 201

      assert_lists_equal(t_data, result.output_t, 1.0e-05)
      assert_nx_lists_equal(x_data, result.output_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - event function with interpolation" do
      nx_true = Nx.tensor(1, type: :u8)

      event_fn = fn _t, x ->
        answer = Nx.less_equal(x[0], Nx.tensor(0.0)) == nx_true
        answer = if answer, do: :halt, else: :continue
        %{status: answer}
      end

      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [event_fn: event_fn]

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 9
      assert result.count_loop__increment_step == 8
      assert result.count_save == 2
      assert result.terminal_event == :halt
      assert result.terminal_output == :continue

      assert length(result.ode_t) == 9
      assert length(result.ode_x) == 9
      assert length(result.output_t) == 33
      assert length(result.output_x) == 33

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_in_delta(last_time, 2.161317515510217, 1.0e-10)

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/x.csv")

      assert_lists_equal(result.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - no data interpolation (refine == 1) together with an output function" do
      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, x -> DummyOutput.add_data(dummy_output_name, %{t: t, x: x}) end

      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [refine: 1, output_fn: output_fn]

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert result.count_save == 2
      assert result.unhandled_termination == true
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 51
      assert length(result.output_x) == 51

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/x.csv")

      # data = result.output_t |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/time_refine_1-elixir.csv", data)

      # data = result.output_x |> Enum.map(fn xn -> "#{Nx.to_number(xn[0])}  #{Nx.to_number(xn[1])}  " end) |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/x_refine_1-elixir.csv", data)
      assert_lists_equal(result.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 51
      assert length(t_data) == 51

      assert_lists_equal(t_data, result.output_t, 1.0e-04)
      assert_nx_lists_equal(x_data, result.output_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - no data interpolation (refine == 1), no caching, output function with terminal output" do
      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, x -> DummyOutput.add_data_and_halt(dummy_output_name, %{t: t, x: x}) end

      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [refine: 1, output_fn: output_fn]

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 1
      assert result.count_loop__increment_step == 1
      assert result.count_save == 2
      assert result.terminal_output == :halt
      assert result.terminal_event == :continue
      assert length(result.ode_t) == 2
      assert length(result.ode_x) == 2
      assert length(result.output_t) == 2
      assert length(result.output_x) == 2

      expected_t = read_csv("test/fixtures/octave_results/van_der_pol/output_fn_halt/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/output_fn_halt/x.csv")

      # data = result.output_t |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/time_refine_1-elixir.csv", data)

      # data = result.output_x |> Enum.map(fn xn -> "#{Nx.to_number(xn[0])}  #{Nx.to_number(xn[1])}  " end) |> Enum.join("\n")
      # File.write!("test/fixtures/integrator/integrator/runge_kutta_45_test/x_refine_1-elixir.csv", data)
      assert_lists_equal(result.output_t, expected_t, 1.0e-04)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 2
      assert length(t_data) == 2

      assert_lists_equal(t_data, result.output_t, 1.0e-04)
      assert_nx_lists_equal(x_data, result.output_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - no caching of results" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)
      opts = [store_resuts?: false]

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.068129

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert result.count_save == 2
      assert result.unhandled_termination == true

      assert Enum.empty?(result.output_t)
      assert Enum.empty?(result.output_x)
      assert Enum.empty?(result.ode_t)
      assert Enum.empty?(result.ode_x)
    end

    test "throws an exception if too many errors" do
      stepper_fn = &DormandPrince45.integrate/5
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &Demo.van_der_pol_fn/2

      t_start = 0.0
      t_end = 20.0
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      # From Octave (or equivalently, from Utils.starting_stepsize/7):
      initial_tstep = 0.007418363820761442

      # Set the max_number_of_errors to 1 so that an exception should be thrown:
      opts = [abs_tol: 1.0e-2, rel_tol: 1.0e-2, max_number_of_errors: 1]

      assert_raise Integrator.MaxErrorsExceededError, "Too many errors", fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts)
      end
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
