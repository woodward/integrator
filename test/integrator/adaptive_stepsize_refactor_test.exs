defmodule Integrator.AdaptiveStepsizeRefactorTest do
  @moduledoc false
  use Integrator.TestCase, async: true
  import Nx, only: :sigils

  alias Integrator.AdaptiveStepsizeRefactor
  alias Integrator.AdaptiveStepsizeRefactor.NxOptions
  alias Integrator.DataCollector
  alias Integrator.ExternalFnAdapter
  alias Integrator.NonLinearEqnRoot
  alias Integrator.Point
  alias Integrator.RungeKutta.BogackiShampine23
  alias Integrator.RungeKutta.DormandPrince45
  alias Integrator.SampleEqns

  describe "integrate" do
    test "works - no data interpolation (refine == 1)" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      ode_fn = &SampleEqns.van_der_pol_fn/2
      order = DormandPrince45.order()

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        refine: 1,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(0.068129)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      assert result.count_cycles__compute_step == Nx.s32(78)
      assert result.count_loop__increment_step == Nx.s32(50)
      assert result.error_count == Nx.s32(0)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/x.csv")

      {output_t, output_x} = DataCollector.get_data(pid) |> Point.split_points_into_t_and_x()
      assert length(output_t) == 51
      assert length(output_x) == 51

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)

      assert result.elapsed_time_μs > 1
    end

    test "works - data interpolation (refine = 4)" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        refine: 4,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(0.068129)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      assert result.count_cycles__compute_step == Nx.s32(78)
      assert result.count_loop__increment_step == Nx.s32(50)

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert length(output_t) == 201
      assert length(output_x) == 201
      assert result.elapsed_time_μs > 1

      # Verify the last time step is correct (bug fix!):
      last_point = points |> List.last()
      assert_all_close(last_point.t, Nx.f64(20.0), atol: 1.0e-10, rtol: 1.0e-10)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - fixed stepsize output that's a tensor with specific values" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn,
        fixed_output_times?: true,
        fixed_output_dt: 1.0
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(6.812920690579614e-02)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      assert result.count_cycles__compute_step == Nx.s32(78)
      assert result.count_loop__increment_step == Nx.s32(50)

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert length(output_t) == 21
      assert length(output_x) == 21
      assert result.elapsed_time_μs > 1

      # # Verify the last time step is correct (bug fix!):
      last_point = points |> List.last()
      assert_in_delta(Nx.to_number(last_point.t), 20.0, 1.0e-10)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-03, rtol: 1.0e-03)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-03, rtol: 1.0e-03)
    end

    test "works - event function with interpolation" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        event_fn: &SampleEqns.falling_particle_event_fn/2,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(0.068129)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      assert result.count_cycles__compute_step == Nx.s32(9)
      assert result.count_loop__increment_step == Nx.s32(8)
      assert result.terminal_event == Nx.u8(0)
      assert result.status_non_linear_eqn_root == Nx.u8(1)
      # assert result.terminal_output == :continue

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      # assert length(result.ode_t) == 9
      # assert length(result.ode_x) == 9
      # assert length(result.output_t) == 33
      # assert length(result.output_x) == 33

      # Verify the last time step is correct (bug fix!):
      last_point = points |> List.last()
      assert_all_close(last_point.t, Nx.f64(2.161317515510217), atol: 1.0e-07, rtol: 1.0e-07)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/x.csv")

      #      actual_t = output_t |> Enum.map(&Nx.to_number(&1)) |> Enum.join("\n")
      #      File.write!("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/junk_actual_t.csv", actual_t)
      #      actual_x = output_x |> Enum.map(fn x -> "#{Nx.to_number(x[0])}    #{Nx.to_number(x[1])}\n" end)
      #      File.write!("test/fixtures/octave_results/van_der_pol/event_fn_positive_x0_only/junk_actual_x.csv", actual_x)

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
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

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      ode_fn = &SampleEqns.van_der_pol_fn/2

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(0.1)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        speed: 1.0,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        refine: 4,
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(6.812920690579614e-02)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      [last_t | _rest] = output_t |> Enum.reverse()

      # This is a 0.1 second simulation, so the elapsed time should be close to 100 ms:
      assert abs(Nx.to_number(result.elapsed_time_μs) / 1000.0 - 100) <= 40

      # write_t(output_t, "test/fixtures/octave_results/van_der_pol/speed/t_elixir2.csv")
      # write_x(output_x, "test/fixtures/octave_results/van_der_pol/speed/x_elixir2.csv")

      # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 0.1, 1.0e-14)

      [last_x | _rest] = output_x |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_x[0]), 1.990933460195306, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -0.172654870547380, 1.0e-13)

      assert result.count_cycles__compute_step == Nx.s32(10)
      assert result.count_loop__increment_step == Nx.s32(10)
      # assert result.terminal_event == :continue
      # assert result.terminal_output == :continue

      # assert length(result.ode_t) == 11
      # assert length(result.ode_x) == 11
      # assert length(result.output_t) == 41
      # assert length(result.output_x) == 41

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-16, rtol: 1.0e-16)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-15, rtol: 1.0e-15)
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

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.1, type: :f64)
      x0 = Nx.tensor([2.0, 0.0], type: :f64)

      opts = [
        speed: 0.5,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.tensor(1.0e-11, type: :f64),
        rel_tol: Nx.tensor(1.0e-11, type: :f64),
        refine: 1,
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(5.054072392284442e-03, type: :f64)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      [last_t | _rest] = output_t |> Enum.reverse()

      # Elapsed time should be something close to 0.1 * 2 or 200 ms:
      assert abs(Nx.to_number(result.elapsed_time_μs) / 1000.0 - 200) <= 60

      # output_t_contents = output_t |> Enum.map(&"#{Nx.to_number(&1)}\n") |> Enum.join()
      # output_x_contents = output_x |> Enum.map(&"#{Nx.to_number(&1[0])}  #{Nx.to_number(&1[1])}\n") |> Enum.join()
      # File.write!("test/fixtures/octave_results/van_der_pol/speed_high_fidelity/junk_t_elixir.csv", output_t_contents)
      # File.write!("test/fixtures/octave_results/van_der_pol/speed_high_fidelity/junk_x_elixir.csv", output_x_contents)

      # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 0.1, 1.0e-14)

      [last_x | _rest] = output_x |> Enum.reverse()

      assert_in_delta(Nx.to_number(last_x[0]), 1.990933460195490, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -0.172654870547865, 1.0e-13)

      # Why do these not match up???
      # assert result.count_cycles__compute_step == Nx.s32(18)
      # assert result.count_loop__increment_step == Nx.s32(18)
      # assert result.terminal_event == :halt
      # assert result.terminal_output == :continue

      # assert length(result.ode_t) == 19
      # assert length(result.ode_x) == 19
      # assert length(result.output_t) == 73
      # assert length(result.output_x) == 73

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/speed_high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/speed_high_fidelity/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
    end

    test "works - high fidelity" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        abs_tol: Nx.f64(1.0e-10),
        rel_tol: Nx.f64(1.0e-10),
        type: :f64,
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(0.007418363820761442)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      assert result.count_cycles__compute_step == Nx.s32(1037)
      assert result.count_loop__increment_step == Nx.s32(1027)

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      # assert length(result.ode_t) == 1028
      # assert length(result.ode_x) == 1028
      # assert length(result.output_t) == 4109
      # assert length(result.output_x) == 4109

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/high_fidelity/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "works - event function with interpolation - ballode - high fidelity - one bounce" do
      event_fn = &SampleEqns.falling_particle_event_fn/2
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      ode_fn = &SampleEqns.falling_particle/2

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(30.0)
      x0 = Nx.f64([0.0, 20.0])

      opts = [
        event_fn: event_fn,
        #  zero_fn: zero_fn,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-14),
        rel_tol: Nx.f64(1.0e-14),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(1.472499532027109e-03)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      [last_t | _rest] = output_t |> Enum.reverse()

      # write_t(result.output_t, "test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/x_elixir.csv")

      # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 4.077471967380223, 1.0e-14)

      [last_x | _rest] = output_x |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_x[0]), 0.0, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -20.0, 1.0e-13)

      assert result.count_cycles__compute_step == Nx.s32(18)
      assert result.count_loop__increment_step == Nx.s32(18)
      assert result.terminal_event == Nx.u8(0)
      assert result.status_non_linear_eqn_root == Nx.u8(1)
      # assert result.terminal_output == :continue

      # assert length(result.ode_t) == 19
      # assert length(result.ode_x) == 19
      # assert length(result.output_t) == 73
      # assert length(result.output_x) == 73

      expected_t = read_nx_list("test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
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

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(0.1)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        refine: 4,
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(6.812920690579614e-02)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      [last_t | _rest] = output_t |> Enum.reverse()

      # write_t(result.output_t, "test/fixtures/octave_results/van_der_pol/speed/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/van_der_pol/speed/x_elixir.csv")

      assert length(output_t) == 41

      # # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 0.1, 1.0e-14)

      [last_x | _rest] = output_x |> Enum.reverse()
      # Expected values are from Octave:
      assert_in_delta(Nx.to_number(last_x[0]), 1.990933460195306, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -0.172654870547380, 1.0e-13)

      assert result.count_cycles__compute_step == Nx.s32(10)
      assert result.count_loop__increment_step == Nx.s32(10)
      assert result.terminal_event == Nx.u8(1)
      # assert result.terminal_output == :continue

      # assert length(ode_t) == 11
      # assert length(ode_x) == 11
      assert length(output_t) == 41
      assert length(output_x) == 41

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/speed/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-16, rtol: 1.0e-16)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-15, rtol: 1.0e-15)
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

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(3.0)
      # t_end = Nx.f64(0.45)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn,
        fixed_output_times?: true,
        fixed_output_dt: Nx.f64(0.05)
      ]

      # t_values = Nx.linspace(t_start, t_end, n: 61) |> Nx.to_list() |> Enum.map(&Nx.f64(&1))

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(6.812920690579614e-02)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert result.count_cycles__compute_step == Nx.s32(10)
      assert result.count_loop__increment_step == Nx.s32(9)
      # assert length(result.ode_t) == 10
      # assert length(result.ode_x) == 10
      # assert length(result.output_t) == 61
      # assert length(result.output_x) == 61

      # Verify the last time step is correct (this check was the result of a bug fix!):
      [last_time | _rest] = output_t |> Enum.reverse()

      # output_t_contents = output_t |> Enum.map(&"#{Nx.to_number(&1)}\n") |> Enum.join()
      # output_x_contents = output_x |> Enum.map(&"#{Nx.to_number(&1[0])}  #{Nx.to_number(&1[1])}\n") |> Enum.join()
      # File.write!("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/junk_t.csv", output_t_contents)
      # File.write!("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/junk_x.csv", output_x_contents)

      assert_in_delta(Nx.to_number(last_time), 3.0, 1.0e-07)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/fixed_stepsize_output_2/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-02, rtol: 1.0e-02)
    end

    test "shows an error status if too many integration errors" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      x0 = Nx.f64([2.0, 0.0])

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(0.007418363820761442)

      # Set the max_number_of_errors to 1 so that an error will bubble up:
      opts = [
        abs_tol: Nx.f64(1.0e-2),
        rel_tol: Nx.f64(1.0e-2),
        max_number_of_errors: 1,
        type: :f64,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      assert result.status_integration == Nx.u8(2)

      points = DataCollector.get_data(pid)

      # This would be length 169 except for the fact that the simulation was terminated early due to the error count:
      assert length(points) == 53

      # This would be 61 except for the fact that the simulation was terminated early due to the error count:
      assert result.count_cycles__compute_step == Nx.s32(16)

      # This would be 42 except for the fact that the simulation was terminated early due to the error count:
      assert result.count_loop__increment_step == Nx.s32(13)
    end

    test "works - uses Bogacki-Shampine23" do
      stepper_fn = &BogackiShampine23.integrate/6
      interpolate_fn = &BogackiShampine23.interpolate/4
      order = BogackiShampine23.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        refine: 4,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(1.778279410038923e-02)

      #  Octave:
      #    fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
      #    [t,y] = ode23 (fvdp, [0, 20], [2, 0], odeset( "Refine", 4));

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert result.count_cycles__compute_step == Nx.s32(189)
      assert result.count_loop__increment_step == Nx.s32(171)
      # assert length(result.ode_t) == 172
      # assert length(result.ode_x) == 172
      # assert length(result.output_t) == 685
      # assert length(result.output_x) == 685

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.tensor(20.0), atol: 1.0e-10, rtol: 1.0e-10)

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-05, rtol: 1.0e-05)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-05, rtol: 1.0e-05)
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

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(0.1)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        refine: 4,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-12),
        rel_tol: Nx.f64(1.0e-12),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(2.020515504676623e-04)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert result.count_cycles__compute_step == Nx.s32(952)
      assert result.count_loop__increment_step == Nx.s32(950)
      # assert length(result.ode_t) == 951
      # assert length(result.ode_x) == 951
      # assert length(result.output_t) == 3_801
      # assert length(result.output_x) == 3_801

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.f64(0.1), atol: 1.0e-11, rtol: 1.0e-11)

      # write_t(output_t, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/t_elixir.csv")
      # write_x(output_x, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/x_elixir.csv")

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_high_fidelity/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
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

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(0.1)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        refine: 1,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-12),
        rel_tol: Nx.f64(1.0e-12),
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(2.020515504676623e-04)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          initial_tstep,
          x0,
          order,
          opts
        )

      points = DataCollector.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert result.count_cycles__compute_step == Nx.s32(952)
      assert result.count_loop__increment_step == Nx.s32(950)
      # assert length(result.ode_t) == 951
      # assert length(result.ode_x) == 951
      # assert length(result.output_t) == 951
      # assert length(result.output_x) == 951

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.f64(0.1), atol: 1.0e-11, rtol: 1.0e-11)

      # write_t(output_t, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/t_elixir.csv")
      # write_x(output_x, "test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/x_elixir.csv")

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/bogacki_shampine_23_hi_fi_no_interpolation/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)
    end
  end

  describe "starting_stepsize" do
    test "works" do
      order = 5
      t0 = 0.0
      x0 = ~VEC[2.0 0.0]f64
      abs_tol = 1.0e-06
      rel_tol = 1.0e-03
      norm_control? = Nx.u8(0)

      starting_stepsize =
        AdaptiveStepsizeRefactor.starting_stepsize(order, &van_der_pol_fn/2, t0, x0, abs_tol, rel_tol, norm_control?)

      assert_all_close(starting_stepsize, Nx.tensor(0.068129, type: :f64), atol: 1.0e-6, rtol: 1.0e-6)
    end

    test "works - high fidelity ballode example to double precision accuracy (works!!!)" do
      order = 5
      t0 = ~VEC[  0.0  ]f64
      x0 = ~VEC[  0.0 20.0  ]f64
      abs_tol = Nx.tensor(1.0e-14, type: :f64)
      rel_tol = Nx.tensor(1.0e-14, type: :f64)
      norm_control? = Nx.u8(0)
      ode_fn = &SampleEqns.falling_particle/2

      starting_stepsize = AdaptiveStepsizeRefactor.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, norm_control?)
      assert_all_close(starting_stepsize, Nx.tensor(0.001472499532027109, type: :f64), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "does NOT work for precision :f16" do
      order = 5
      t0 = ~VEC[  0.0  ]f16
      x0 = ~VEC[  2.0  0.0  ]f16
      abs_tol = Nx.tensor(1.0e-06, type: :f16)
      rel_tol = Nx.tensor(1.0e-03, type: :f16)
      norm_control? = Nx.u8(0)
      ode_fn = &SampleEqns.van_der_pol_fn/2

      starting_stepsize = AdaptiveStepsizeRefactor.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, norm_control?)

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

  describe "convert_to_nx_options" do
    test "uses the defaults from nimble options (and defaults for machine_eps and tolerance in the type specified)" do
      use_default_opts = []
      t_start = 0.0
      t_end = 10.0
      order = 5

      nx_options = AdaptiveStepsizeRefactor.convert_to_nx_options(t_start, t_end, order, use_default_opts)
      assert %NxOptions{} = nx_options

      assert nx_options.type == {:f, 32}
      assert nx_options.max_number_of_errors == Nx.s32(5_000)
      assert nx_options.max_step == Nx.f32(1.0)
      assert nx_options.refine == 4
      assert nx_options.speed == Nx.Constants.infinity(:f32)
      assert nx_options.fixed_output_times? == Nx.u8(0)
      assert nx_options.fixed_output_dt == Nx.f32(0.0)
      assert nx_options.order == 5
      assert nx_options.norm_control? == Nx.u8(1)
      assert nx_options.abs_tol == Nx.f32(1.0e-06)
      assert nx_options.rel_tol == Nx.f32(1.0e-03)
      assert nx_options.nx_while_loop_integration? == Nx.u8(1)

      assert nx_options.event_fn_adapter == %ExternalFnAdapter{external_fn: &Integrator.ExternalFnAdapter.no_op_double_arity_fn/2}

      assert nx_options.output_fn_adapter == %ExternalFnAdapter{}
      assert nx_options.output_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)

      assert nx_options.zero_fn_adapter == %ExternalFnAdapter{}
      assert nx_options.zero_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)

      # --------------------------------------
      # Values are passed on to NonLinearEqnRoot.NxOptions:
      non_linear_eqn_root_nx_options = nx_options.non_linear_eqn_root_nx_options
      assert %NonLinearEqnRoot.NxOptions{} = non_linear_eqn_root_nx_options

      assert_all_close(non_linear_eqn_root_nx_options.machine_eps, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.machine_eps) == {:f, 64}

      assert_all_close(non_linear_eqn_root_nx_options.tolerance, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.tolerance) == {:f, 64}

      assert non_linear_eqn_root_nx_options.type == {:f, 64}
      assert non_linear_eqn_root_nx_options.max_iterations == 1_000
      assert non_linear_eqn_root_nx_options.max_fn_eval_count == 1_000
      assert non_linear_eqn_root_nx_options.output_fn_adapter == %ExternalFnAdapter{}
      assert non_linear_eqn_root_nx_options.output_fn_adapter.external_fn == (&Integrator.ExternalFnAdapter.no_op_fn/1)
    end

    test "works and does not blow up if t_start and t_end are tensors, not floats" do
      use_default_opts = []
      t_start = Nx.f32(0.0)
      t_end = Nx.f32(10.0)
      order = 5

      nx_options = AdaptiveStepsizeRefactor.convert_to_nx_options(t_start, t_end, order, use_default_opts)
      assert %NxOptions{} = nx_options

      assert nx_options.max_step == Nx.f32(1.0)
    end

    test "sets :refine to 1 if using fixed sizes, regardless of the value" do
      opts = [refine: 4, fixed_output_times?: true]
      t_start = Nx.f32(0.0)
      t_end = Nx.f32(10.0)
      order = 5

      nx_options = AdaptiveStepsizeRefactor.convert_to_nx_options(t_start, t_end, order, opts)
      assert %NxOptions{} = nx_options

      assert nx_options.refine == 1
    end

    test "overrides the defaults if provided, including those in the NonLinearEqnRoot.NxOptions" do
      opts = [
        type: :f64,
        max_number_of_errors: 2,
        max_step: 3.0,
        refine: 3,
        speed: 0.5,
        fixed_output_times?: true,
        fixed_output_dt: 0.5,
        norm_control?: false,
        abs_tol: 1.0e-08,
        rel_tol: 1.0e-04,
        output_fn: &Math.sin/1,
        event_fn: &Kernel.max/2,
        zero_fn: &Kernel.min/2,
        #
        # NonLinearEqnRoot.NxOptions:
        machine_eps: 1.0e-03,
        tolerance: 1.0e-04,
        max_iterations: 3,
        max_fn_eval_count: 4,
        nonlinear_eqn_root_output_fn: &Math.sin/1
      ]

      t_start = 0.0
      t_end = 10.0
      order = 3

      nx_options = AdaptiveStepsizeRefactor.convert_to_nx_options(t_start, t_end, order, opts)
      assert %NxOptions{} = nx_options

      assert nx_options.type == {:f, 64}
      assert nx_options.max_number_of_errors == Nx.s32(2)
      assert nx_options.max_step == Nx.f64(3.0)
      assert nx_options.refine == 1
      assert nx_options.speed == Nx.f64(0.5)
      assert nx_options.fixed_output_times? == Nx.u8(1)
      assert nx_options.fixed_output_dt == Nx.f64(0.5)
      assert nx_options.order == 3
      assert nx_options.norm_control? == Nx.u8(0)
      assert nx_options.abs_tol == Nx.f64(1.0e-08)
      assert nx_options.rel_tol == Nx.f64(1.0e-04)
      assert nx_options.nx_while_loop_integration? == Nx.u8(0)

      assert nx_options.event_fn_adapter == %ExternalFnAdapter{external_fn: &:erlang.max/2}
      assert nx_options.output_fn_adapter == %ExternalFnAdapter{external_fn: &Math.sin/1}
      assert nx_options.zero_fn_adapter == %ExternalFnAdapter{external_fn: &:erlang.min/2}

      # --------------------------------------
      # Values are passed on to NonLinearEqnRoot.NxOptions:
      non_linear_eqn_root_nx_options = nx_options.non_linear_eqn_root_nx_options
      assert %NonLinearEqnRoot.NxOptions{} = non_linear_eqn_root_nx_options

      assert_all_close(non_linear_eqn_root_nx_options.machine_eps, Nx.f64(1.0e-03), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.machine_eps) == {:f, 64}

      assert_all_close(non_linear_eqn_root_nx_options.tolerance, Nx.f64(1.0e-04), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(non_linear_eqn_root_nx_options.tolerance) == {:f, 64}

      assert non_linear_eqn_root_nx_options.type == {:f, 64}
      assert non_linear_eqn_root_nx_options.max_iterations == 3
      assert non_linear_eqn_root_nx_options.max_fn_eval_count == 4
      assert non_linear_eqn_root_nx_options.output_fn_adapter == %ExternalFnAdapter{external_fn: &Math.sin/1}
    end
  end
end
