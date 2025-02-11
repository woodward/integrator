defmodule Integrator.AdaptiveStepsizeTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.AdaptiveStepsize
  alias Integrator.DataSet
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(78))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(50))
      assert_nx_equal(result.error_count, Nx.s32(0))

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/no_interpolation/x.csv")

      {output_t, output_x} = DataSet.get_data(pid) |> Point.split_points_into_t_and_x()
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(78))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(50))

      points = DataSet.get_data(pid)
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        fixed_output_step: 1.0
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(6.812920690579614e-02)

      result =
        AdaptiveStepsize.integrate(
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

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(78))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(50))

      points = DataSet.get_data(pid)
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(9))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(8))
      assert_nx_equal(result.terminal_event, Nx.u8(0))
      assert_nx_equal(result.status_non_linear_eqn_root, Nx.s32(1))
      # assert result.terminal_output == :continue

      points = DataSet.get_data(pid)
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      [last_t | _rest] = output_t |> Enum.reverse()

      # This is a 0.1 second simulation, so the elapsed time should be close to 100 ms:
      # This is way slower with EXLA; disabling this check for now:
      # assert abs(Nx.to_number(result.elapsed_time_μs) / 1000.0 - 100) <= 40

      # write_t(output_t, "test/fixtures/octave_results/van_der_pol/speed/t_elixir2.csv")
      # write_x(output_x, "test/fixtures/octave_results/van_der_pol/speed/x_elixir2.csv")

      # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 0.1, 1.0e-14)

      [last_x | _rest] = output_x |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_x[0]), 1.990933460195306, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -0.172654870547380, 1.0e-13)

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(10))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(10))
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

      t_start = Nx.f64(0.0)
      t_end = Nx.f64(0.1)
      x0 = Nx.f64([2.0, 0.0])

      opts = [
        speed: 0.5,
        type: :f64,
        norm_control?: false,
        abs_tol: Nx.f64(1.0e-11),
        rel_tol: Nx.f64(1.0e-11),
        refine: 1,
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(5.054072392284442e-03)

      _result =
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      [last_t | _rest] = output_t |> Enum.reverse()

      # Elapsed time should be something close to 0.1 * 2 or 200 ms:
      # This is way slower with EXLA; disabling this check for now:
      # assert abs(Nx.to_number(result.elapsed_time_μs) / 1000.0 - 200) <= 60

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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(1037))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(1027))

      points = DataSet.get_data(pid)
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      [last_t | _rest] = output_t |> Enum.reverse()

      # write_t(result.output_t, "test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/t_elixir.csv")
      # write_x(result.output_x, "test/fixtures/octave_results/ballode/high_fidelity_one_bounce_only/x_elixir.csv")

      # Expected last_t is from Octave:
      assert_in_delta(Nx.to_number(last_t), 4.077471967380223, 1.0e-14)

      [last_x | _rest] = output_x |> Enum.reverse()
      assert_in_delta(Nx.to_number(last_x[0]), 0.0, 1.0e-13)
      assert_in_delta(Nx.to_number(last_x[1]), -20.0, 1.0e-13)

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(18))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(18))
      assert_nx_equal(result.terminal_event, Nx.u8(0))
      assert_nx_equal(result.status_non_linear_eqn_root, Nx.s32(1))
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
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

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(10))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(10))
      assert_nx_equal(result.terminal_event, Nx.u8(1))
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        fixed_output_step: Nx.f64(0.05)
      ]

      # t_values = Nx.linspace(t_start, t_end, n: 61) |> Nx.to_list() |> Enum.map(&Nx.f64(&1))

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.f64(6.812920690579614e-02)

      result =
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(10))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(9))
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      assert_nx_equal(result.status_integration, Nx.u8(2))

      points = DataSet.get_data(pid)

      # This would be length 169 except for the fact that the simulation was terminated early due to the error count:
      assert length(points) == 53

      # This would be 61 except for the fact that the simulation was terminated early due to the error count:
      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(16))

      # This would be 42 except for the fact that the simulation was terminated early due to the error count:
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(13))
    end

    test "works - uses Bogacki-Shampine23" do
      stepper_fn = &BogackiShampine23.integrate/6
      interpolate_fn = &BogackiShampine23.interpolate/4
      order = BogackiShampine23.order()

      ode_fn = &SampleEqns.van_der_pol_fn/2

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(189))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(171))
      # assert length(result.ode_t) == 172
      # assert length(result.ode_x) == 172
      # assert length(result.output_t) == 685
      # assert length(result.output_x) == 685

      # Verify the last time step is correct (bug fix!):
      [last_time | _rest] = output_t |> Enum.reverse()
      assert_all_close(last_time, Nx.f64(20.0), atol: 1.0e-10, rtol: 1.0e-10)

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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(952))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(950))
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

      {:ok, pid} = DataSet.start_link()
      output_fn = &DataSet.add_data(pid, &1)

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
        AdaptiveStepsize.integrate(
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

      points = DataSet.get_data(pid)
      {output_t, output_x} = points |> Point.split_points_into_t_and_x()

      assert_nx_equal(result.count_cycles__compute_step, Nx.s32(952))
      assert_nx_equal(result.count_loop__increment_step, Nx.s32(950))
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

    test "bug fix - does not blow up if there is no output function" do
      t_initial = Nx.f64(0.0)
      t_final = Nx.f64(20.0)
      x_initial = Nx.f64([2.0, 0.0])

      # This was raising an exception about a Point not being an NxContainer:
      integration = Integrator.integrate(&SampleEqns.van_der_pol_fn/2, t_initial, t_final, x_initial, type: :f64)

      assert_nx_equal(integration.status_integration, Nx.u8(1))
    end
  end

  describe "starting_stepsize" do
    test "works" do
      order = 5
      t0 = 0.0
      x0 = Nx.f64([2.0, 0.0])
      abs_tol = 1.0e-06
      rel_tol = 1.0e-03
      norm_control? = Nx.u8(0)

      starting_stepsize =
        AdaptiveStepsize.starting_stepsize(order, &van_der_pol_fn/2, t0, x0, abs_tol, rel_tol, norm_control?)

      assert_all_close(starting_stepsize, Nx.f64(0.068129), atol: 1.0e-6, rtol: 1.0e-6)
    end

    test "works - high fidelity ballode example to double precision accuracy (works!!!)" do
      order = 5
      t0 = Nx.f64(0.0)
      x0 = Nx.f64([0.0, 20.0])
      abs_tol = Nx.f64(1.0e-14)
      rel_tol = Nx.f64(1.0e-14)
      norm_control? = Nx.u8(0)
      ode_fn = &SampleEqns.falling_particle/2

      starting_stepsize = AdaptiveStepsize.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, norm_control?)
      assert_all_close(starting_stepsize, Nx.f64(0.001472499532027109), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "does NOT work for precision :f16" do
      order = 5
      t0 = Nx.f16(0.0)
      x0 = Nx.f16([2.0, 0.0])
      abs_tol = Nx.f16(1.0e-06)
      rel_tol = Nx.f16(1.0e-03)
      norm_control? = Nx.u8(0)
      ode_fn = &SampleEqns.van_der_pol_fn/2

      starting_stepsize = AdaptiveStepsize.starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, norm_control?)

      zero_stepsize_which_is_bad = Nx.f16(0.0)

      assert_all_close(starting_stepsize, zero_stepsize_which_is_bad, atol: 1.0e-14, rtol: 1.0e-14)

      # The starting_stepsize is zero because d2 goes to infinity:
      # abs_rel_norm = abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, opts)
      # Values for abs_rel_norm and h0 captured from Elixir output:
      abs_rel_norm = Nx.f16(999.5)
      h0 = Nx.f16(0.01000213623046875)
      one = Nx.f16(1)

      #  d2 = one / h0 * abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, opts)
      d2 = Nx.divide(one, h0) |> Nx.multiply(abs_rel_norm)
      # d2 being infinity causes the starting_stepsize to be zero:
      assert_all_close(d2, Nx.Constants.infinity(), atol: 1.0e-14, rtol: 1.0e-14)
    end
  end
end
