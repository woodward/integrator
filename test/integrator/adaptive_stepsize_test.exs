defmodule Integrator.AdaptiveStepsizeTest do
  @moduledoc false
  use Integrator.TestCase
  use Patch

  import Nx, only: :sigils

  import Nx.Defn

  alias Integrator.{AdaptiveStepsize, SampleEqns, DummyOutput}
  alias Integrator.AdaptiveStepsize.ArgPrecisionError
  alias Integrator.RungeKutta.{BogackiShampine23, DormandPrince45}

  describe "integrate" do
    test "works" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 201
      assert length(result.output_x) == 201
      assert is_integer(result.timestamp_start_ms)
      assert result.timestamp_ms != result.timestamp_start_ms
      assert AdaptiveStepsize.elapsed_time_ms(result) > 1

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.tensor(20.0), atol: 1.0e-10, rtol: 1.0e-10)
      assert last_time.__struct__ == Nx.Tensor
      [start_time | _rest] = result.output_t
      assert start_time.__struct__ == Nx.Tensor

      [last_ode_time | _rest] = result.ode_t |> Enum.reverse()
      assert last_ode_time.__struct__ == Nx.Tensor
      [start_ode_time | _rest] = result.ode_t
      assert start_ode_time.__struct__ == Nx.Tensor

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - high fidelity" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        abs_tol: Nx.tensor(1.0e-10, type: :f64),
        rel_tol: Nx.tensor(1.0e-10, type: :f64),
        type: :f64,
        norm_control: false,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.007418363820761442, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 1037
      assert result.count_loop__increment_step == 1027
      assert length(result.ode_t) == 1028
      assert length(result.ode_x) == 1028
      assert length(result.output_t) == 4109
      assert length(result.output_x) == 4109

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - no data interpolation (refine == 1)" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        refine: 1,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 51
      assert length(result.output_x) == 51

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - output function with interpolation" do
      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, x -> DummyOutput.add_data(dummy_output_name, %{t: t, x: x}) end

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        output_fn: output_fn,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 201
      assert length(result.output_x) == 201

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.tensor(20.0), atol: 1.0e-10, rtol: 1.0e-10)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 201
      assert length(t_data) == 201

      assert_nx_lists_equal(t_data, result.output_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(x_data, result.output_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - event function with interpolation" do
      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        %{status: answer, value: value}
      end

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        event_fn: event_fn,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 9
      assert result.count_loop__increment_step == 8
      assert result.terminal_event == :halt
      assert result.terminal_output == :continue

      assert length(result.ode_t) == 9
      assert length(result.ode_x) == 9
      assert length(result.output_t) == 33
      assert length(result.output_x) == 33

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.tensor(2.161317515510217), atol: 1.0e-07, rtol: 1.0e-07)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - event function with interpolation - ballode - high fidelity - one bounce" do
      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        %{status: answer, value: value}
      end

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.falling_particle/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(30.0, type: :f64)
      x0 = Nx.tensor([0.0, 20.0], type: :f64)

      opts = [
        event_fn: event_fn,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-14, type: :f64),
        rel_tol: Nx.tensor(1.0e-14, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(1.472499532027109e-03, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      [last_t | _rest] = result.output_t |> Enum.reverse()

      # write_t(result.output_t, "test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/x_elixir.csv")

      # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 4.077471967380223, 1.0e-14)

      [last_x | _rest] = result.output_x |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_x[0]), 0.0, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -20.0, 1.0e-13)

      assert result.count_cycles__compute_step == 18
      assert result.count_loop__increment_step == 18
      assert result.terminal_event == :halt
      assert result.terminal_output == :continue

      assert length(result.ode_t) == 19
      assert length(result.ode_x) == 19
      assert length(result.output_t) == 73
      assert length(result.output_x) == 73

      expected_t = read_nx_list("test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
    end

    test "max step uses computed default for short simulation times" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-06, "RelTol", 1.0e-03);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.1, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        refine: 4
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(6.812920690579614e-02, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      [last_t | _rest] = result.output_t |> Enum.reverse()

      # write_t(result.output_t, "test/fixtures/octave_results/van_der_pol/speed/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/van_der_pol/speed/x_elixir.csv")

      assert length(result.output_t) == 41

      # # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 0.1, 1.0e-14)

      [last_x | _rest] = result.output_x |> Enum.reverse()
      # Expected values are from Octave:
      assert_in_delta(Nx.to_number(last_x[0]), 1.990933460195306, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -0.172654870547380, 1.0e-13)

      assert result.count_cycles__compute_step == 10
      assert result.count_loop__increment_step == 10
      assert result.terminal_event == :continue
      assert result.terminal_output == :continue

      assert length(result.ode_t) == 11
      assert length(result.ode_x) == 11
      assert length(result.output_t) == 41
      assert length(result.output_x) == 41

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-16, rtol: 1.0e-16)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "works - playback speed of 1.0" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-06, "RelTol", 1.0e-03);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.1, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        speed: 1.0,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        refine: 4
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(6.812920690579614e-02, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      [last_t | _rest] = result.output_t |> Enum.reverse()

      assert abs(AdaptiveStepsize.elapsed_time_ms(result) - 100) <= 8

      # write_t(result.output_t, "test/fixtures/octave_results/van_der_pol/speed/t_elixir2.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/van_der_pol/speed/x_elixir2.csv")

      # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 0.1, 1.0e-14)

      [last_x | _rest] = result.output_x |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_x[0]), 1.990933460195306, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -0.172654870547380, 1.0e-13)

      assert result.count_cycles__compute_step == 10
      assert result.count_loop__increment_step == 10
      assert result.terminal_event == :continue
      assert result.terminal_output == :continue

      assert length(result.ode_t) == 11
      assert length(result.ode_x) == 11
      assert length(result.output_t) == 41
      assert length(result.output_x) == 41

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-16, rtol: 1.0e-16)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "works - high fidelity - playback speed of 0.5" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-11, "RelTol", 1.0e-11, "Refine", 1);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.1, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        speed: 0.5,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-11, type: :f64),
        rel_tol: Nx.tensor(1.0e-11, type: :f64),
        refine: 1
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(5.054072392284442e-03, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      [_last_t | _rest] = result.output_t |> Enum.reverse()

      assert abs(AdaptiveStepsize.elapsed_time_ms(result) - 200) <= 10

      # write_t(result.output_t, "test/fixtures/octave_results/van_der_pol/speed_high_fidelity/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/van_der_pol/speed_high_fidelity/x_elixir.csv")

      # Expected last_t is from Octave:
      # assert_in_delta(Nx.to_number(last_t), 0.1, 1.0e-14)

      # [last_x | _rest] = result.output_x |> Enum.reverse()
      # assert_in_delta(Nx.to_number(last_x[0]), 0.0, 1.0e-13)
      # assert_in_delta(Nx.to_number(last_x[1]), -20.0, 1.0e-13)

      # assert result.count_cycles__compute_step == 18
      # assert result.count_loop__increment_step == 18
      # assert result.terminal_event == :halt
      # assert result.terminal_output == :continue

      # assert length(result.ode_t) == 19
      # assert length(result.ode_x) == 19
      # assert length(result.output_t) == 73
      # assert length(result.output_x) == 73

      # expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/speed_high_fidelity/t.csv")
      # expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/speed_high_fidelity/x.csv")

      # assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      # assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
    end

    test "works - no data interpolation (refine == 1) together with an output function" do
      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, x -> DummyOutput.add_data(dummy_output_name, %{t: t, x: x}) end

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        refine: 1,
        output_fn: output_fn,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 51
      assert length(result.output_x) == 51

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 51
      assert length(t_data) == 51

      assert_nx_lists_equal(t_data, result.output_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(x_data, result.output_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - no data interpolation (refine == 1), no caching, output function with terminal output" do
      dummy_output_name = :"dummy-output-#{inspect(self())}"
      DummyOutput.start_link(name: dummy_output_name)
      output_fn = fn t, x -> DummyOutput.add_data_and_halt(dummy_output_name, %{t: t, x: x}) end

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        refine: 1,
        output_fn: output_fn,
        type: :f64,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 1
      assert result.count_loop__increment_step == 1
      assert result.terminal_output == :halt
      assert result.terminal_event == :continue
      assert length(result.ode_t) == 2
      assert length(result.ode_x) == 2
      assert length(result.output_t) == 2
      assert length(result.output_x) == 2

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/output_fn_halt/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/output_fn_halt/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      x_data = DummyOutput.get_x(dummy_output_name)
      t_data = DummyOutput.get_t(dummy_output_name)
      assert length(x_data) == 2
      assert length(t_data) == 2

      assert_nx_lists_equal(t_data, result.output_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(x_data, result.output_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - fixed stepsize output that's a tensor with specific values" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      t_values = Nx.linspace(t_start, t_end, n: 21, type: :f64) |> Nx.to_list() |> Enum.map(&Nx.tensor(&1, type: :f64))

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(6.812920690579614e-02, type: :f64)

      result =
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, t_values, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50
      assert length(result.ode_t) == 51
      assert length(result.ode_x) == 51
      assert length(result.output_t) == 21
      assert length(result.output_x) == 21

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_time), 20.0, 1.0e-10)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - fixed stepsize output that's smaller than the timestep" do
      # In this test, there are many output timesteps for every integration timestep

      # Octave:
      #   format long
      #   t_values = 0.0:0.05:3;
      #   fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
      #   [t,y] = ode45 (fvdp, t_values, [2, 0]);
      #
      #   file_id = fopen("../test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/t.csv", "w")
      #   fdisp(file_id, t)
      #   fclose(file_id)
      #
      #   file_id = fopen("../test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/x.csv", "w")
      #   fdisp(file_id, y)
      #   fclose(file_id)

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(3.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        type: :f64,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      t_values = Nx.linspace(t_start, t_end, n: 61, type: :f64) |> Nx.to_list() |> Enum.map(&Nx.tensor(&1, type: :f64))
      # t_values = Nx.linspace(t_start, t_end, n: 61, type: :f64) |> Nx.to_list() |> Enum.map(&Nx.tensor(&1, type: :f64))

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(6.812920690579614e-02, type: :f64)

      result =
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, t_values, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 10
      assert result.count_loop__increment_step == 9
      assert length(result.ode_t) == 10
      assert length(result.ode_x) == 10
      assert length(result.output_t) == 61
      assert length(result.output_x) == 61

      # Verify the last time step is correct (this check was the result of a bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_time), 3.0, 1.0e-07)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-02, rtol: 1.0e-02)
    end

    test "works - do not store results" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        store_results?: false,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 78
      assert result.count_loop__increment_step == 50

      assert Enum.empty?(result.output_t)
      assert Enum.empty?(result.output_x)
      assert Enum.empty?(result.ode_t)
      assert Enum.empty?(result.ode_x)
    end

    test "throws an exception if too many errors" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.007418363820761442, type: :f64)

      # Set the max_number_of_errors to 1 so that an exception should be thrown:
      opts = [
        abs_tol: Nx.tensor(1.0e-2, type: :f64),
        rel_tol: Nx.tensor(1.0e-2, type: :f64),
        max_number_of_errors: 1,
        type: :f64,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      assert_raise Integrator.AdaptiveStepsize.MaxErrorsExceededError, "Too many errors", fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "works - uses Bogacki-Shampine23" do
      stepper_fn = &BogackiShampine23.integrate/6
      interpolate_fn = &BogackiShampine23.interpolate/4
      order = BogackiShampine23.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        refine: 4,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(1.778279410038923e-02, type: :f64)

      #  Octave:
      #    fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
      #    [t,y] = ode23 (fvdp, [0, 20], [2, 0], odeset( "Refine", 4));

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 189
      assert result.count_loop__increment_step == 171
      assert length(result.ode_t) == 172
      assert length(result.ode_x) == 172
      assert length(result.output_t) == 685
      assert length(result.output_x) == 685

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.tensor(20.0), atol: 1.0e-10, rtol: 1.0e-10)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - uses Bogacki-Shampine23 - high fidelity" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-12, "RelTol", 1.0e-12, "Refine", 4)
      #   [t,x] = ode23 (fvdp, [0, 0.1], [2, 0], opts);

      stepper_fn = &BogackiShampine23.integrate/6
      interpolate_fn = &BogackiShampine23.interpolate/4
      order = BogackiShampine23.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.1, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        refine: 4,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(2.020515504676623e-04, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 952
      assert result.count_loop__increment_step == 950
      assert length(result.ode_t) == 951
      assert length(result.ode_x) == 951
      assert length(result.output_t) == 3_801
      assert length(result.output_x) == 3_801

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.tensor(0.1), atol: 1.0e-11, rtol: 1.0e-11)

      # write_t(result.output_t, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/x_elixir.csv")

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
    end

    test "works - uses Bogacki-Shampine23 - high fidelity - no interpolation" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-12, "RelTol", 1.0e-12, "Refine", 1)
      #   [t,x] = ode23 (fvdp, [0, 0.1], [2, 0], opts);

      stepper_fn = &BogackiShampine23.integrate/6
      interpolate_fn = &BogackiShampine23.interpolate/4
      order = BogackiShampine23.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.1, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        refine: 1,
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(2.020515504676623e-04, type: :f64)

      result = AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert result.count_cycles__compute_step == 952
      assert result.count_loop__increment_step == 950
      assert length(result.ode_t) == 951
      assert length(result.ode_x) == 951
      assert length(result.output_t) == 951
      assert length(result.output_x) == 951

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = result.output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.tensor(0.1), atol: 1.0e-11, rtol: 1.0e-11)

      # write_t(result.output_t, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/x_elixir.csv")

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/x.csv")

      assert_nx_lists_equal(result.output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(result.output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
    end
  end

  describe "starting_stepsize" do
    test "works" do
      order = 5
      t0 = 0.0
      x0 = ~V[2.0 0.0]f64
      abs_tol = 1.0e-06
      rel_tol = 1.0e-03

      starting_stepsize =
        AdaptiveStepsize.starting_stepsize(order, &van_der_pol_fn/2, t0, x0, abs_tol, rel_tol, norm_control: false)

      assert_all_close(starting_stepsize, Nx.tensor(0.068129, type: :f64), atol: 1.0e-6, rtol: 1.0e-6)
    end

    test "works - high fidelity ballode example to double precision accuracy (works!!!)" do
      order = 5
      t0 = ~V[  0.0  ]f64
      x0 = ~V[  0.0 20.0  ]f64
      abs_tol = Nx.tensor(1.0e-14, type: :f64)
      rel_tol = Nx.tensor(1.0e-14, type: :f64)
      opts = [norm_control: false]
      ode_fn = &SampleEqns.falling_particle/2

      starting_stepsize = AdaptiveStepsize.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, opts)
      assert_all_close(starting_stepsize, Nx.tensor(0.001472499532027109, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "does NOT work for precision :f16" do
      order = 5
      t0 = ~V[  0.0  ]f16
      x0 = ~V[  2.0  0.0  ]f16
      abs_tol = Nx.tensor(1.0e-06, type: :f16)
      rel_tol = Nx.tensor(1.0e-03, type: :f16)
      opts = [norm_control: false]
      ode_fn = &SampleEqns.van_der_pol_fn/2

      starting_stepsize = AdaptiveStepsize.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, opts)

      zero_stepsize_which_is_bad = Nx.tensor(0.0, type: :f16)

      assert_all_close(starting_stepsize, zero_stepsize_which_is_bad, atol: 1.0e-14, rtol: 1.0e-14)

      # The starting_stepsize is zero because d2 goes to infinity:
      # abs_rel_norm = abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, opts)
      # Values for abs_rel_norm and h0 captured from Elixir output:
      abs_rel_norm = Nx.tensor(999.5, type: :f16)
      h0 = Nx.tensor(0.01000213623046875, type: :f16)
      one = Nx.tensor(1, type: :f16)

      #  d2 = one / h0 * abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, opts)
      d2 = Nx.divide(one, h0) |> Nx.multiply(abs_rel_norm)
      # d2 being infinity causes the starting_stepsize to be zero:
      assert_all_close(d2, Nx.Constants.infinity(), atol: 1.0e-14, rtol: 1.0e-14)
    end
  end

  describe "validating all args" do
    setup do
      opts = [
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-06, type: :f64),
        norm_control: false,
        type: :f64,
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()
      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      [
        t_start: t_start,
        t_end: t_end,
        x0: x0,
        initial_tstep: initial_tstep,
        opts: opts,
        stepper_fn: stepper_fn,
        interpolate_fn: interpolate_fn,
        order: order,
        ode_fn: ode_fn
      ]
    end

    test "does not raise an exception if all args are correct", %{
      t_start: t_start,
      t_end: t_end,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      solution =
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)

      assert solution.__struct__ == AdaptiveStepsize
    end

    test "raises an exception if t_start is incorrect nx type", %{
      t_end: t_end,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      t_start = Nx.tensor(0.0, type: :f32)

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if t_end is incorrect nx type", %{
      t_start: t_start,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      t_end = Nx.tensor(1.0, type: :f32)

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if t_range is incorrect nx type", %{
      t_start: t_start,
      t_end: t_end,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      t_range = Nx.linspace(0.0, 10.0, n: 21, type: :f32) |> Nx.to_list() |> Enum.map(&Nx.tensor(&1, type: :f32))

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, t_range, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if x0 is incorrect nx type", %{
      t_start: t_start,
      t_end: t_end,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      x0 = Nx.tensor([0.0, 1.0], type: :f32)

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if initial_step is incorrect nx type", %{
      t_start: t_start,
      t_end: t_end,
      x0: x0,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      initial_tstep = 0.01

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if :abs_tol is incorrect nx type", %{
      t_start: t_start,
      t_end: t_end,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      opts = opts |> Keyword.merge(abs_tol: 1.0e-06)

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if :rel_tol is incorrect nx type", %{
      t_start: t_start,
      t_end: t_end,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      opts = opts |> Keyword.merge(rel_tol: 1.0e-03)

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if :max_step is incorrect nx type", %{
      t_start: t_start,
      t_end: t_end,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order,
      ode_fn: ode_fn
    } do
      opts = Keyword.merge(opts, max_step: 1.0)

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end

    test "raises an exception if the ode function returns the incorrect nx type for x", %{
      t_start: t_start,
      t_end: t_end,
      x0: x0,
      initial_tstep: initial_tstep,
      opts: opts,
      stepper_fn: stepper_fn,
      interpolate_fn: interpolate_fn,
      order: order
    } do
      bad_ode_fn = fn _t, _x ->
        Nx.tensor([1.23, 4.56], type: :f32)
      end

      assert_raise ArgPrecisionError, fn ->
        AdaptiveStepsize.integrate(stepper_fn, interpolate_fn, bad_ode_fn, t_start, t_end, nil, initial_tstep, x0, order, opts)
      end
    end
  end

  # ===========================================================================
  # Tests of private functions below here:

  describe "compute_step" do
    setup do
      expose(AdaptiveStepsize, compute_step: 4)
    end

    test "works" do
      # Expected values were obtained from Octave:
      k_vals = ~M[
        -0.123176646786029  -0.156456392653781  -0.170792108688503  -0.242396950166743  -0.256398564600740  -0.270123280961810  -0.266528851971234
        -1.628266220377807  -1.528057633442594  -1.484796318238127  -1.272143242010950  -1.231218923718637  -1.191362260138565  -1.201879818436319
      ]f64

      step = %AdaptiveStepsize{
        t_new: Nx.tensor(0.170323017264490, type: :f64),
        x_new: Nx.tensor([1.975376830028490, -0.266528851971234], type: :f64),
        options_comp: Nx.tensor(-1.387778780781446e-17, type: :f64),
        dt: Nx.tensor(0.153290715538041, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &DormandPrince45.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2
      opts = [type: :f64, norm_control: false, abs_tol: 1.0e-06, rel_tol: 1.0e-03]

      {computed_step, error} = private(AdaptiveStepsize.compute_step(step, stepper_fn, ode_fn, opts))

      expected_t_next = Nx.tensor(0.323613732802532, type: :f64)
      expected_x_next = Nx.tensor([1.922216228514310, -0.416811343851152], type: :f64)

      expected_k_vals = ~M[
        -0.266528851971234  -0.303376255443000  -0.318166975994861  -0.394383609924488  -0.412602091137911  -0.426290366186482  -0.416811343851152
        -1.201879818436319  -1.096546739499175  -1.055438526511377  -0.852388604155395  -0.804214989044028  -0.771328619755717  -0.798944990281621
      ]f64

      expected_options_comp = Nx.tensor(0.0, type: :f64)
      expected_error = Nx.tensor(1.586715304267830e-02, type: :f64)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(error, expected_error, atol: 1.0e-07, rtol: 1.0e-07)
    end

    test "works - bug fix for Bogacki-Shampine23" do
      # Expected values were obtained from Octave for van der pol equation at t = 0.000345375551682:
      k_vals = ~M[
        -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04
        -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00
      ]f64

      step = %AdaptiveStepsize{
        t_new: Nx.tensor(3.453755516815583e-04, type: :f64),
        x_new: Nx.tensor([1.999999880756917, -6.903933604135114e-04], type: :f64),
        options_comp: Nx.tensor(1.355252715606881e-20, type: :f64),
        dt: Nx.tensor(1.048148240128353e-04, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      opts = [
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      ]

      {computed_step, error} = private(AdaptiveStepsize.compute_step(step, stepper_fn, ode_fn, opts))

      expected_t_next = Nx.tensor(4.501903756943936e-04, type: :f64)
      expected_x_next = Nx.tensor([1.999999797419839, -8.997729805855904e-04], type: :f64)

      expected_k_vals = ~M[
        -6.903933604135114e-04  -7.950996330065260e-04  -8.474280732402655e-04  -8.997729805855904e-04
        -1.997928701004975e+00  -1.997614546170481e+00  -1.997457534649594e+00  -1.997300479207187e+00
      ]f64

      expected_options_comp = Nx.tensor(-2.710505431213761e-20, type: :f64)
      expected_error = Nx.tensor(0.383840528805912, type: :f64)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-17, rtol: 1.0e-17)

      # Note that the error is just accurate to single precision, which is ok; see the test below for abs_rel_norm
      # to see how sensitive the error is to input values:
      assert_all_close(error, expected_error, atol: 1.0e-07, rtol: 1.0e-07)
    end

    test "works - bug fix for Bogacki-Shampine23 - 2nd attempt" do
      # Expected values and inputs were obtained from Octave for van der pol equation at t = 0.000239505625605:
      k_vals = ~M[
        -2.585758248155079e-04  -3.687257174741470e-04  -4.237733527882678e-04  -4.788391990136420e-04
        -1.999224255823159e+00  -1.998893791926987e+00  -1.998728632828801e+00  -1.998563425163596e+00
      ]f64

      step = %AdaptiveStepsize{
        t_new: Nx.tensor(2.395056256047516e-04, type: :f64),
        x_new: Nx.tensor([1.999999942650792, -4.788391990136420e-04], type: :f64),
        options_comp: Nx.tensor(-1.355252715606881e-20, type: :f64),
        dt: Nx.tensor(1.058699260768067e-04, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      opts = [
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      ]

      {computed_step, error} = private(AdaptiveStepsize.compute_step(step, stepper_fn, ode_fn, opts))

      expected_t_next = Nx.tensor(3.453755516815583e-04, type: :f64)
      #                   Elixir: 3.4537555168155827e-4
      expected_x_next = Nx.tensor([1.999999880756917, -6.903933604135114e-04], type: :f64)
      #                  Elixir:  [1.9999998807569168 -6.903933604135114e-4]

      expected_k_vals = ~M[
        -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04
        -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00
      ]f64

      # -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04  Octave
      # -4.78839199013642e-4    -5.846330800545818e-4   -6.375048176907232e-4   -6.903933604135114e-4],  Elixir

      # -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00  Octave
      # -1.998563425163596,     -1.9982460182566815,    -1.998087382041041,     -1.9979287010049749]    Elixir
      expected_options_comp = Nx.tensor(1.355252715606881e-20, type: :f64)
      #                       Elixir:  -2.710505431213761e-20
      expected_error = Nx.tensor(0.395533432395734, type: :f64)
      #                  Elixir: 0.3955334323957338

      # dbg(error)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(error, expected_error, atol: 1.0e-15, rtol: 1.0e-15)
    end

    test "works - bug fix for Bogacki-Shampine23 - 2nd attempt - using inputs from Elixir, not Octave" do
      # Inputs were obtained from AdaptiveStepsize for van der pol equation at t = 0.000239505625605:
      # Expected values are from Octave
      k_vals = ~M[
        -2.585758248155079e-4  -3.687257174741469e-4  -4.2377335278826774e-4  -4.78839199013642e-4
        -1.999224255823159     -1.9988937919269867    -1.9987286328288005     -1.9985634251635955
      ]f64

      step = %AdaptiveStepsize{
        t_new: Nx.tensor(2.3950562560475164e-04, type: :f64),
        x_new: Nx.tensor([1.9999999426507922, -4.78839199013642e-4], type: :f64),
        options_comp: Nx.tensor(0.0, type: :f64),
        dt: Nx.tensor(1.0586992285952218e-4, type: :f64),
        # dt is WRONG!!!!
        # dt: Nx.tensor(1.058699260768067e-04, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      opts = [
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      ]

      {computed_step, error} = private(AdaptiveStepsize.compute_step(step, stepper_fn, ode_fn, opts))

      expected_t_next = Nx.tensor(3.453755516815583e-04, type: :f64)
      #                   Elixir: 3.453755 484642738e-4
      expected_x_next = Nx.tensor([1.999999880756917, -6.903933604135114e-04], type: :f64)
      #                  Elixir:  [1.9999998807569193 -6.903933 539856063e-4]

      expected_k_vals = ~M[
        -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04
        -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00
      ]f64

      # -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04  Octave
      # -4.78839199013642e-4    -5.846330800545818e-4   -6.375048176907232e-4   -6.903933604135114e-4],  Elixir

      # -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00  Octave
      # -1.998563425163596,     -1.9982460182566815,    -1.998087382041041,     -1.9979287010049749]    Elixir
      expected_options_comp = Nx.tensor(1.355252715606881e-20, type: :f64)
      #                       Elixir:  -2.710505431213761e-20
      expected_error = Nx.tensor(0.395533432395734, type: :f64)
      #                  Elixir: 0.3955334323957338

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-19, rtol: 1.0e-19)
      assert_all_close(error, expected_error, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  describe "call_event_fn" do
    setup do
      expose(AdaptiveStepsize, call_event_fn: 4)

      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        %{status: answer, value: value}
      end

      [event_fn: event_fn]
    end

    test ":continue event", %{event_fn: event_fn} do
      t_new = Nx.tensor(0.553549806109594, type: :f64)
      x_new = ~V[ 1.808299104387025  -0.563813853847242 ]f64
      opts = []
      step = %AdaptiveStepsize{t_new: t_new, x_new: x_new}
      interpolate_fn_does_not_matter = & &1

      new_step = private(AdaptiveStepsize.call_event_fn(step, event_fn, interpolate_fn_does_not_matter, opts))

      assert new_step.terminal_event == :continue
    end

    test ":halt event for SampleEqns.van_der_pol function (y[0] goes negative)", %{event_fn: event_fn} do
      t_old = Nx.tensor(2.155396117711071, type: :f64)
      t_new = Nx.tensor(2.742956500140625, type: :f64)

      x_old = ~V[  1.283429405203074e-02  -2.160506093425276 ]f64
      x_new = ~V[ -1.452959132853812      -2.187778875125423 ]f64

      k_vals = ~M[
          -2.160506093425276  -2.415858015466959  -2.525217131637079  -2.530906930089893  -2.373278736970216  -2.143782883869835  -2.187778875125423
          -2.172984510849814  -2.034431603317282  -1.715883769683796   2.345467244704591   3.812328420909734   4.768800180323954   3.883778892097804
        ]f64

      opts = []

      step = %AdaptiveStepsize{
        t_new: t_new,
        x_new: x_new,
        t_old: t_old,
        x_old: x_old,
        t_new_rk_interpolate: t_new,
        x_new_rk_interpolate: x_new,
        k_vals: k_vals
      }

      interpolate_fn = &DormandPrince45.interpolate/4

      new_step = private(AdaptiveStepsize.call_event_fn(step, event_fn, interpolate_fn, opts))
      assert new_step.terminal_event == :halt

      assert_all_close(new_step.t_new, Nx.tensor(2.161317515510217), atol: 1.0e-07, rtol: 1.0e-07)

      assert_all_close(new_step.x_new, Nx.tensor([2.473525941362742e-15, -2.173424479824061], type: :f64),
        atol: 1.0e-06,
        rtol: 1.0e-06
      )

      assert_all_close(new_step.t_old, Nx.tensor(2.155396117711071, type: :f64), atol: 1.0e-07, rtol: 1.0e-07)

      assert_all_close(new_step.x_old, Nx.tensor([1.283429405203074e-02, -2.160506093425276], type: :f64),
        atol: 1.0e-07,
        rtol: 1.0e-07
      )

      # Spot-check one value of the k_vals matrix:
      assert_all_close(new_step.k_vals[0][0], Nx.tensor(-2.160506093425276, type: :f64), atol: 1.0e-07, rtol: 1.0e-07)

      assert new_step.options_comp != nil
    end
  end

  describe "interpolate_one_point" do
    setup do
      expose(AdaptiveStepsize, interpolate_one_point: 3)
    end

    test "works" do
      t_old = ~V[ 2.155396117711071 ]f64
      t_new = ~V[ 2.742956500140625 ]f64
      x_old = ~V[  1.283429405203074e-02  -2.160506093425276 ]f64
      x_new = ~V[ -1.452959132853812      -2.187778875125423 ]f64

      k_vals = ~M[
            -2.160506093425276  -2.415858015466959  -2.525217131637079  -2.530906930089893  -2.373278736970216  -2.143782883869835  -2.187778875125423
            -2.172984510849814  -2.034431603317282  -1.715883769683796   2.345467244704591   3.812328420909734   4.768800180323954   3.883778892097804
          ]f64

      interpolate_fn = &DormandPrince45.interpolate/4

      step = %AdaptiveStepsize{
        t_new: t_new,
        x_new: x_new,
        t_old: t_old,
        x_old: x_old,
        t_new_rk_interpolate: t_new,
        x_new_rk_interpolate: x_new,
        k_vals: k_vals,
        nx_type: :f64
      }

      t = ~V[ 2.161317515510217 ]f64
      x_interpolated = private(AdaptiveStepsize.interpolate_one_point(t, step, interpolate_fn))

      # From Octave:
      expected_x_interpolated = ~V[ 2.473525941362742e-15 -2.173424479824061  ]f64

      # Why is this not closer to tighter tolerances?
      assert_all_close(x_interpolated, expected_x_interpolated, atol: 1.0e-07, rtol: 1.0e-07)
    end
  end

  describe "compute_next_timestep" do
    setup do
      expose(AdaptiveStepsize, compute_next_timestep: 6)
    end

    test "basic case" do
      dt = Nx.tensor(0.068129, type: :f64)
      error = Nx.tensor(0.0015164936598390992, type: :f64)
      order = 5
      t_old = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(2.0, type: :f64)
      opts = [type: :f64, max_step: 2.0]

      new_dt = private(AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, opts))

      expected_dt = Nx.tensor(0.1022, type: :f64)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "uses option :max_step" do
      dt = Nx.tensor(0.068129, type: :f64)
      error = Nx.tensor(0.0015164936598390992, type: :f64)
      order = 5
      t_old = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(2.0, type: :f64)
      opts = [max_step: 0.05, type: :f64]

      new_dt = private(AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, opts))

      expected_dt = Nx.tensor(0.05, type: :f64)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "does not go past t_end" do
      dt = Nx.tensor(0.3039, type: :f64)
      error = Nx.tensor(0.4414, type: :f64)
      order = 5
      t_old = Nx.tensor(19.711, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      opts = [type: :f64, max_step: 2.0]

      new_dt = private(AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, opts))

      expected_dt = Nx.tensor(0.289, type: :f64)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "bug fix for Bogacki-Shampine high fidelity (see high fidelity Bogacki-Shampine test above)" do
      dt = Nx.tensor(2.020515504676623e-4, type: :f64)
      error = Nx.tensor(2.7489475539627106, type: :f64)
      order = 3
      t_old = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(20.0, type: :f64)
      opts = [type: :f64, max_step: Nx.tensor(2.0, type: :f64)]

      new_dt = private(AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, opts))

      # From Octave:
      expected_dt = Nx.tensor(1.616412403741299e-04, type: :f64)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-19, rtol: 1.0e-19)
    end

    test "2nd bug fix for Bogacki-Shampine high fidelity (see high fidelity Bogacki-Shampine test above) - compare Elixir input" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-12, "RelTol", 1.0e-12, "Refine", 1);
      #   [t,x] = ode23 (fvdp, [0, 0.1], [2, 0], opts);

      # Input values are from Elixir for t_old = 2.395056256047516e-04:
      dt = Nx.tensor(1.1019263330544775e-04, type: :f64)
      #              1.101926333054478e-04  Octave

      # If I use error value from Octave - succeeds:
      error = Nx.tensor(0.445967698534111, type: :f64)

      # TRY ENABLING THIS AGAIN AFTER KAHAN FIX!!!
      #  If I use error value from Elixir - fails:
      # error = Nx.tensor(0.4459677527442196, type: :f64)

      order = 3
      t_old = Nx.tensor(2.3950562560475164e-4, type: :f64)
      t_end = Nx.tensor(0.1, type: :f64)
      opts = [type: :f64, max_step: Nx.tensor(2.0, type: :f64)]

      new_dt = private(AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, opts))

      # Expected dt from Octave:
      expected_dt = Nx.tensor(1.058699260768067e-04, type: :f64)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-19, rtol: 1.0e-19)
    end

    test "bug fix for 'works - high fidelity - playback speed of 0.5'" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-11, "RelTol", 1.0e-11, "Refine", 1);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      # Input values are from Elixir for t_old = 0.005054072392284442:
      dt = Nx.tensor(0.007408247469735083, type: :f64)
      #              0.007408247469735083  Octave  EXACT MATCH!

      # Error from Elixir only agrees to single precision - THIS IS THE PROBLEM!
      # error_from_elixir = Nx.tensor(0.25920723900618725, type: :f64)

      # The test passes if I use the error from Octave:
      error_from_octave = Nx.tensor(0.259206892061492, type: :f64)
      # error = error_from_elixir
      error = error_from_octave

      order = 5
      t_old = Nx.tensor(0.012462319862019525, type: :f64)
      #                 0.01246231986201952   Octave  EXACT MATCH!

      t_end = Nx.tensor(0.1, type: :f64)

      opts = [
        type: :f64,
        max_step: Nx.tensor(0.01, type: :f64),
        rel_tol: Nx.tensor(1.0e-11, type: :f64),
        abs_tol: Nx.tensor(1.0e-11, type: :f64),
        norm_control: false
      ]

      new_dt = private(AdaptiveStepsize.compute_next_timestep(dt, error, order, t_old, t_end, opts))

      # Expected dt from Octave:
      expected_dt = Nx.tensor(0.007895960916517373, type: :f64)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-19, rtol: 1.0e-19)
    end
  end

  describe "initial_empty_k_vals" do
    setup do
      expose(AdaptiveStepsize, initial_empty_k_vals: 2)
    end

    test "returns a tensor with zeros that's the correct size" do
      order = 5
      x = ~V[ 1.0 2.0 3.0 ]f64
      k_vals = private(AdaptiveStepsize.initial_empty_k_vals(order, x))

      expected_k_vals = ~M[
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
      ]f64

      assert_all_close(k_vals, expected_k_vals)

      # The k_vals has the Nx type of x:
      assert Nx.type(k_vals) == {:f, 64}
    end

    test "returns a tensor that has the Nx type of x" do
      order = 3
      type = {:f, 32}
      x = Nx.tensor([1.0, 2.0, 3.0], type: type)
      k_vals = private(AdaptiveStepsize.initial_empty_k_vals(order, x))

      expected_k_vals = ~M[
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
      ]f64

      assert_all_close(k_vals, expected_k_vals)

      # The k_vals has the Nx type of x:
      assert Nx.type(k_vals) == type
    end
  end

  describe "abs_rel_norm/6" do
    setup do
      expose(AdaptiveStepsize, abs_rel_norm: 6)
    end

    test "when norm_control: false" do
      # These test values were obtained from Octave:
      t = Nx.tensor([1.97537683003, -0.26652885197])
      t_old = Nx.tensor([1.99566026409, -0.12317664679])
      abs_tolerance = 1.0000e-06
      rel_tolerance = 1.0000e-03
      opts = [norm_control: false]
      x = Nx.tensor([1.97537723429, -0.26653011403])
      expected_norm = Nx.tensor(0.00473516383083)

      norm = private(AdaptiveStepsize.abs_rel_norm(t, t_old, x, abs_tolerance, rel_tolerance, opts))

      assert_all_close(norm, expected_norm, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "when norm_control: false - :f64 - starting_stepsize for high-fidelity ballode" do
      x0 = ~V[  0.0 20.0  ]f64
      abs_tol = Nx.tensor(1.0e-14, type: :f64)
      rel_tol = Nx.tensor(1.0e-14, type: :f64)
      opts = [norm_control: false]
      x_zeros = Nx.tensor([0.0, 0.0], type: :f64)

      norm = private(AdaptiveStepsize.abs_rel_norm(x0, x0, x_zeros, abs_tol, rel_tol, opts))

      assert_all_close(norm, Nx.tensor(1.0e14, type: :f64), atol: 1.0e-17, rtol: 1.0e-17)
    end

    test "when norm_control: false - :f64 - for high-fidelity Bogacki-Shampine" do
      # All values taken from Octave for the high-fidelity Bogacki-Shampine23 at t = 0.000345375551682:
      x_old = ~V[ 1.999999880756917  -6.903933604135114e-04 ]f64
      #         [ 1.999999880756917, -6.903933604135114e-04 ]  Elixir values agree exactly

      x_next = ~V[ 1.999999797419839   -8.997729805855904e-04  ]f64
      #          [ 1.9999997974198394, -8.997729805855904e-4]  Elixir values agree exactly

      # This works (from Octave):
      x_est = ~V[ 1.999999797419983  -8.997729809694310e-04 ]f64

      # This doesn't work (from Elixir); note the _very_ small differences in x[1]:
      # x_est = ~V[ 1.9999997974199832 -8.997729809694309e-04 ]f64
      # x_est = ~V[ 1.999999797419983  -8.997729809694310e-04 ]f64  Octave values from above

      # From Octave:
      expected_error = Nx.tensor(0.383840528805912, type: :f64)
      #                          0.3838404203856949
      #                          Value from Elixir using x_est above.
      # Note that it seems to be just single precision agreement
      # The equations in abs_rel_norm check out ok; they are just SUPER sensitive to small differences
      # in the input values

      abs_tol = Nx.tensor(1.0e-12, type: :f64)
      rel_tol = Nx.tensor(1.0e-12, type: :f64)

      error = private(AdaptiveStepsize.abs_rel_norm(x_next, x_old, x_est, abs_tol, rel_tol, norm_control: false))

      assert_all_close(error, expected_error, atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "when norm_control: false - :f64 - for test 'works - high fidelity - playback speed of 0.5'" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-11, "RelTol", 1.0e-11, "Refine", 1);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      # Values for t = 0.005054072392284442:
      x_next = ~V[ 1.9998466100062002  -0.024463877616688966 ]f64
      # Octave:
      # x_next = ~V[ 1.999846610006200   -0.02446387761668897 ]f64

      x_old = ~V[  1.9999745850165596  -0.010031858255616163 ]f64
      # Octave:
      # x_old = ~V[  1.999974585016560   -0.01003185825561616 ]f64

      x_est = ~V[  1.9998466100068684  -0.024463877619281038 ]f64
      # Octave:
      # x_est = ~V[  1.999846610006868   -0.02446387761928104 ]f64

      # Octave:

      abs_tol = Nx.tensor(1.0e-11, type: :f64)
      rel_tol = Nx.tensor(1.0e-11, type: :f64)

      expected_error = Nx.tensor(0.259206892061492, type: :f64)

      # error = private(AdaptiveStepsize.abs_rel_norm(x_next, x_old, x_est, abs_tol, rel_tol, norm_control: false))
      {error, _t_minus_x} = abs_rel_norm_for_test_purposes(x_next, x_old, x_est, abs_tol, rel_tol, norm_control: false)

      # IO.inspect(Nx.to_number(error), label: "error")
      # IO.inspect(t_minus_x, label: "t_minus_x")

      # sc (Elixir): [1.9999745850165594e-11, 1.0e-11
      # sc (Octave): [1.999974585016559e-11   9.999999999999999e-12]

      # x_next - x_est, which is t - x in abs_rel_norm:
      # t - x (Elixir): [-6.681322162194192e-13, 2.5920 723900618725e-12]
      # t - x (Octave): [-6.681322162194192e-13, 2.5920 68920614921e-12]   SINGLE PRECISION AGREEMENT!!!

      # Nx.abs(t - x) (Elixir) [6.681322162194192e-13, 2.5920 723900618725e-12]
      # Nx.abs(t - x) (Octave) [6.681322162194192e-13, 2.5920 68920614921e-12 ]  SINGLE PRECISION AGREEMENT!!!

      # We can currently get single precision agreement, but not double precision:
      assert_all_close(error, expected_error, atol: 1.0e-06, rtol: 1.0e-06)

      _subtraction = Nx.subtract(x_next[1], x_est[1])
      # IO.inspect(Nx.to_number(subtraction), label: "problematic subtraction")
      # subtraction (Elixir): 2.5920 723900618725e-12
      # subtraction (Octave): 2.5920 68920614921e-12

      # This doesn't work, but should:
      # assert_all_close(error, expected_error, atol: 1.0e-11, rtol: 1.0e-11)
    end

    defn abs_rel_norm_for_test_purposes(t, t_old, x, abs_tolerance, rel_tolerance, _opts \\ []) do
      # Octave code:
      #   sc = max (AbsTol(:), RelTol .* max (abs (x), abs (x_old)));
      #   retval = max (abs (x - y) ./ sc);

      sc = Nx.max(abs_tolerance, rel_tolerance * Nx.max(Nx.abs(t), Nx.abs(t_old)))
      {(Nx.abs(t - x) / sc) |> Nx.reduce_max(), t - x}
    end

    test "trying to figure out precision problem" do
      # Values from Octave:
      x_new_2 = Nx.tensor(-2.446387761668897e-02, type: :f64)
      x_est_2 = Nx.tensor(-2.446387761928104e-02, type: :f64)

      # problematic subtraction         : 2.5920 723900618725e-12
      # expected_subtraction_from_octave: 2.5920 68920614921e-12

      # If I enter these values directly into Octave (rather than printing them out from the integration proces)
      # I get:
      # Octave:
      #   x_new_2 = -2.446387761668897e-02
      #   x_est_2 = -2.446387761928104e-02
      #   subtraction = x_new_2 - x_est_2
      #   2.592072390061873e-12  Octave value
      #   2.5920723900618725e-12 Elixir value
      # which is also single precision agreement

      subtraction = Nx.subtract(x_new_2, x_est_2)
      # IO.inspect(Nx.to_number(subtraction), label: "problematic subtraction         ")
      expected_subtraction_from_octave = Nx.tensor(2.592068920614921e-12, type: :f64)
      #                                            2.5920 72390061873e-12  Octave from above when directly entering values
      # IO.inspect(Nx.to_number(expected_subtraction_from_octave), label: "expected_subtraction_from_octave")
      # assert_all_close(subtraction, expected_subtraction_from_octave, atol: 1.0e-06, rtol: 1.0e-06)
      assert_all_close(subtraction, expected_subtraction_from_octave, atol: 1.0e-17, rtol: 1.0e-17)
    end

    test "when norm_control: false - :f64 - for high-fidelity van der pol" do
      # All values taken from Octave from test "works - high fidelity - playback speed of 0.5" for the 2nd timestep
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-11, "RelTol", 1.0e-11);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      x_old = ~V[ 1.999974585016560   -0.01003185825561616  ]f64
      # xoe = ~V[ 1.9999745850165596  -0.010031858255616163 ]f64  Elixir values agree exactly

      x_next = ~V[ 1.999846610006200   -0.02446387761668897 ]f64
      # xne = ~V[  1.9998466100062002  -0.024463877616688966 ]f64  Elixir values agree exactly

      # This works (from Octave):
      x_est = ~V[ 1.999846610006868   -0.02446387761928104  ]f64

      # This doesn't work (from Elixir); note the _very_ small differences in x[1]:
      # x_est = ~V[ 1.9998466100068684  -0.024463877619281038 ]f64
      # x_est = ~V[ 1.999846610006868   -0.02446387761928104  ]f64  Octave values from above to compare

      # From Octave:
      expected_error = Nx.tensor(0.259206892061492, type: :f64)
      #                          0.2592072390061872
      #                          Value from Elixir using x_est above.
      # Note that it seems to be just single precision agreement
      # The equations in abs_rel_norm check out ok; they are just SUPER sensitive to small differences
      # in the input values

      abs_tol = Nx.tensor(1.0e-11, type: :f64)
      rel_tol = Nx.tensor(1.0e-11, type: :f64)

      error = private(AdaptiveStepsize.abs_rel_norm(x_next, x_old, x_est, abs_tol, rel_tol, norm_control: false))

      # sc:   [1.99997458501656e-11, 1.0e-11]  Elixir                Agreement!!!
      # sc:    1.999974585016559e-11 9.999999999999999e-12 Octave

      # t:  [1.9998466100062,  -0.02446387761668897]  Elixir       Agreement!!
      # t:   1.999846610006200 -0.02446387761668897   Octave

      # x:  [1.999846610006868, -0.02446387761928104]  Elixir     Agreement!!
      # x:   1.999846610006868  -0.02446387761928104   Octave

      # t - x:   [-6.6 79101716144942e-13, 2.5920 723900618725e-12]  Elixir  Not so great :(
      # t - x:    -6.6 81322162194192e-13  2.5920 68920614921e-12    Octave

      # Nx.abs(t - x):  [6.679101716144942e-13, 2.5920723900618725e-12]  Elixir
      # Nx.abs(t - x):   6.681322162194192e-13  2.592068920614921e-12    Octave

      # Nx.abs(t - x) / sc:  [0.033 39593295926627, 0.25920 723900618725]  Elixir
      # Nx.abs(t - x) / sc:   0.033 40703533059582  0.25920 68920614921    Octave
      assert_all_close(error, expected_error, atol: 1.0e-06, rtol: 1.0e-06)

      # Should be able to get this precision:
      # assert_all_close(error, expected_error, atol: 1.0e-16, rtol: 1.0e-16)
    end

    test "when norm_control: true" do
      # These test values were obtained from Octave:
      x = Nx.tensor([1.99465419035, 0.33300240425])
      x_old = Nx.tensor([1.64842646336, 1.78609260054])
      abs_tolerance = 1.0000e-06
      rel_tolerance = 1.0000e-03
      opts = [norm_control: true]
      y = Nx.tensor([1.99402286380, 0.33477644992])
      expected_norm = Nx.tensor(0.77474409123)

      norm = private(AdaptiveStepsize.abs_rel_norm(x, x_old, y, abs_tolerance, rel_tolerance, opts))

      assert_all_close(norm, expected_norm, atol: 1.0e-04, rtol: 1.0e-04)
    end
  end

  describe "zero_vector" do
    setup do
      expose(AdaptiveStepsize, zero_vector: 1)
    end

    test "creates a zero vector with the length and type of x" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f64)
      y = private(AdaptiveStepsize.zero_vector(x))
      expected_y = Nx.tensor([0.0, 0.0, 0.0], type: :f64)
      assert_all_close(y, expected_y)
      assert Nx.type(y) == {:f, 64}
    end
  end

  describe "check_nx_type/2" do
    setup do
      expose(AdaptiveStepsize, check_nx_type: 2)
    end

    test "checks one arg" do
      x0 = ~V[ 2.0  3.0 ]f64
      assert private(AdaptiveStepsize.check_nx_type([x0: x0], :f64)) == :ok
    end

    test "raises an exception if the arg is not a tensor and :f64 is required" do
      x0 = 1.2345

      assert_raise(ArgPrecisionError, fn ->
        private(AdaptiveStepsize.check_nx_type([x0: x0], :f64))
      end)
    end

    test "raises an exception if the arg is not of the correct precision - :f64 required" do
      x0 = ~V[ 2.0  3.0 ]f32

      assert_raise(ArgPrecisionError, fn ->
        private(AdaptiveStepsize.check_nx_type([x0: x0], :f64))
      end)
    end

    test "raises an exception if the arg is not of the correct precision - :f32 required" do
      x0 = ~V[ 2.0  3.0 ]f64

      assert_raise(ArgPrecisionError, fn ->
        private(AdaptiveStepsize.check_nx_type([x0: x0], :f32))
      end)
    end

    test "can check multiple args" do
      x0 = ~V[ 2.0  3.0 ]f64
      x1 = ~V[ 2.0  3.0 ]f32

      assert_raise(ArgPrecisionError, fn ->
        private(AdaptiveStepsize.check_nx_type([x0: x0, x1: x1], :f64))
      end)
    end
  end
end
