defmodule Integrator.AdaptiveStepsizeTest do
  @moduledoc false
  use Integrator.TestCase

  import Nx, only: :sigils

  alias Integrator.AdaptiveStepsize
  alias Integrator.AdaptiveStepsize.ArgPrecisionError
  alias Integrator.DummyOutput
  alias Integrator.RungeKutta.BogackiShampine23
  alias Integrator.RungeKutta.DormandPrince45
  alias Integrator.SampleEqns

  describe "integrate" do
    @tag transferred_to_refactor?: false
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
      assert is_integer(result.timestamp_start_μs)
      assert result.timestamp_μs != result.timestamp_start_μs
      assert AdaptiveStepsize.elapsed_time_μs(result) > 1

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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
    test "works - event function with interpolation" do
      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        {answer, value}
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

    @tag transferred_to_refactor?: false
    test "works - event function with interpolation - ballode - high fidelity - one bounce" do
      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        {answer, value}
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

      assert abs(AdaptiveStepsize.elapsed_time_μs(result) / 1000.0 - 100) <= 40

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

    @tag transferred_to_refactor?: false
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

      assert abs(AdaptiveStepsize.elapsed_time_μs(result) / 1000.0 - 200) <= 10

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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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
    @tag transferred_to_refactor?: false
    test "works" do
      order = 5
      t0 = 0.0
      x0 = ~VEC[2.0 0.0]f64
      abs_tol = 1.0e-06
      rel_tol = 1.0e-03

      starting_stepsize =
        AdaptiveStepsize.starting_stepsize(order, &van_der_pol_fn/2, t0, x0, abs_tol, rel_tol, norm_control: false)

      assert_all_close(starting_stepsize, Nx.tensor(0.068129, type: :f64), atol: 1.0e-6, rtol: 1.0e-6)
    end

    @tag transferred_to_refactor?: false
    test "works - high fidelity ballode example to double precision accuracy (works!!!)" do
      order = 5
      t0 = ~VEC[  0.0  ]f64
      x0 = ~VEC[  0.0 20.0  ]f64
      abs_tol = Nx.tensor(1.0e-14, type: :f64)
      rel_tol = Nx.tensor(1.0e-14, type: :f64)
      opts = [norm_control: false]
      ode_fn = &SampleEqns.falling_particle/2

      starting_stepsize = AdaptiveStepsize.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, opts)
      assert_all_close(starting_stepsize, Nx.tensor(0.001472499532027109, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end

    @tag transferred_to_refactor?: false
    test "does NOT work for precision :f16" do
      order = 5
      t0 = ~VEC[  0.0  ]f16
      x0 = ~VEC[  2.0  0.0  ]f16
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

    @tag transferred_to_refactor?: false
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

  describe "stack" do
    # This function should move somewhere soon
    @tag transferred_to_refactor?: false
    test "puts the arguments in the form required by interpolate/4" do
      # This struct will become a %RungeKuttaStep{} soon:
      step = %{
        t_old: Nx.f32(1),
        t_new_rk_interpolate: Nx.f32(2),
        x_old: Nx.f32([10, 20]),
        x_new_rk_interpolate: Nx.f32([30, 40])
      }

      {t, x} = AdaptiveStepsize.stack(step)

      expected_t = Nx.f32([1, 2])
      expected_x = ~MAT[
        10  30
        20  40
      ]f32

      assert t == expected_t
      assert x == expected_x
    end
  end
end
