defmodule Integrator.IntegrationTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.NxOptions
  alias Integrator.DataSink
  alias Integrator.Point
  alias Integrator.Integration

  describe "can start up and run a simulation" do
    setup do
      initial_x = Nx.f64([2.0, 0.0])
      t_initial = Nx.f64(0.0)
      t_final = Nx.f64(20.0)

      [initial_x: initial_x, t_initial: t_initial, t_final: t_final]
    end

    test "performs the integration via run_async/1 (i.e., a genserver cast starts the run)", %{
      initial_x: initial_x,
      t_initial: t_initial,
      t_final: t_final
    } do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, data_pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(data_pid, self(), &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      {:ok, pid} = Integration.start_link(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      assert Integration.get_status(pid) == :initialized

      Integration.run_async(pid)
      assert Integration.get_status(pid) == :running

      # Change from Process.sleep() to something that periodicaly checks whether the integration is done
      Process.sleep(1_000)

      assert Integration.get_status(pid) == :completed

      {output_t, output_x} = DataSink.get_data(data_pid, pid) |> Point.split_points_into_t_and_x()

      assert length(output_t) == 201
      assert length(output_x) == 201

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "performs the integration via run/1 (i.e., a genserver call starts the run)", %{
      initial_x: initial_x,
      t_initial: t_initial,
      t_final: t_final
    } do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, data_pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(data_pid, self(), &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      {:ok, pid} = Integration.start_link(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      assert Integration.get_status(pid) == :initialized

      :ok = Integration.run(pid)
      assert Integration.get_status(pid) == :completed

      {output_t, output_x} = DataSink.get_data(data_pid, pid) |> Point.split_points_into_t_and_x()

      # actual_t = output_t |> Enum.map(&Nx.to_number(&1)) |> Enum.join("\n")
      # File.write!("test/fixtures/octave_results/van_der_pol/default/junk_actual_t.csv", actual_t)
      # actual_x = output_x |> Enum.map(fn x -> "#{Nx.to_number(x[0])}    #{Nx.to_number(x[1])}\n" end)
      # File.write!("test/fixtures/octave_results/van_der_pol/default/junk_actual_x.csv", actual_x)

      assert length(output_t) == 201
      assert length(output_x) == 201

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "can name the integration genserver", %{
      initial_x: initial_x,
      t_initial: t_initial,
      t_final: t_final
    } do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, data_pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(data_pid, self(), &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn,
        name: :van_der_pol
      ]

      {:ok, pid} = Integration.start_link(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      assert Integration.get_status(:van_der_pol) == :initialized

      :ok = Integration.run(:van_der_pol)
      assert Integration.get_status(:van_der_pol) == :completed

      {output_t, output_x} = DataSink.get_data(data_pid, pid) |> Point.split_points_into_t_and_x()

      assert length(output_t) == 201
      assert length(output_x) == 201

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "can store the data in the genserver itself as well as the data collector", %{
      initial_x: initial_x,
      t_initial: t_initial,
      t_final: t_final
    } do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, data_pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(data_pid, self(), &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn,
        store_data_in_genserver?: true
      ]

      {:ok, pid} = Integration.start_link(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      :ok = Integration.run(pid)

      {output_t, output_x} = DataSink.get_data(data_pid, pid) |> Point.split_points_into_t_and_x()

      # actual_t = output_t |> Enum.map(&Nx.to_number(&1)) |> Enum.join("\n")
      # File.write!("test/fixtures/octave_results/van_der_pol/default/junk_actual_t.csv", actual_t)
      # actual_x = output_x |> Enum.map(fn x -> "#{Nx.to_number(x[0])}    #{Nx.to_number(x[1])}\n" end)
      # File.write!("test/fixtures/octave_results/van_der_pol/default/junk_actual_x.csv", actual_x)

      assert length(output_t) == 201
      assert length(output_x) == 201

      {genserver_output_t, genserver_output_x} = Integration.get_data(pid, pid) |> Point.split_points_into_t_and_x()

      assert length(genserver_output_t) == 201
      assert length(genserver_output_x) == 201
    end

    test "can get the integration step and options from the genserver", %{
      initial_x: initial_x,
      t_initial: t_initial,
      t_final: t_final
    } do
      opts = [type: :f64]
      {:ok, pid} = Integration.start_link(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)

      step = Integration.get_step(pid)
      assert match?(%IntegrationStep{}, step)

      options = Integration.get_options(pid)
      assert match?(%NxOptions{}, options)
    end

    test "can manually step through the integration", %{
      initial_x: initial_x,
      t_initial: t_initial,
      t_final: t_final
    } do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, data_pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(data_pid, self(), &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      {:ok, pid} = Integration.start_link(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      assert Integration.get_status(pid) == :initialized

      number_of_steps =
        1..1_000
        |> Enum.reduce_while(0, fn _i, i_step ->
          if Integration.can_continue_stepping?(pid) do
            {:ok, _step} = Integration.step(pid)
            assert Integration.get_status(pid) == :paused
            {:cont, i_step + 1}
          else
            {:halt, i_step}
          end
        end)

      assert Integration.get_status(pid) == :paused

      assert number_of_steps == 78

      {output_t, output_x} = DataSink.get_data(data_pid, pid) |> Point.split_points_into_t_and_x()

      # actual_t = output_t |> Enum.map(&Nx.to_number(&1)) |> Enum.join("\n")
      # File.write!("test/fixtures/octave_results/van_der_pol/default/junk_actual_t.csv", actual_t)
      # actual_x = output_x |> Enum.map(fn x -> "#{Nx.to_number(x[0])}    #{Nx.to_number(x[1])}\n" end)
      # File.write!("test/fixtures/octave_results/van_der_pol/default/junk_actual_x.csv", actual_x)

      assert length(output_t) == 201
      assert length(output_x) == 201

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "can pause and continue a simulation", %{
      initial_x: initial_x,
      t_initial: t_initial,
      t_final: t_final
    } do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, data_pid} = DataSink.start_link()
      output_fn = &DataSink.add_data(data_pid, self(), &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      {:ok, pid} = Integration.start_link(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)
      assert Integration.get_status(pid) == :initialized

      Integration.run_async(pid)
      assert Integration.get_status(pid) == :running

      Integration.pause(pid)
      assert Integration.get_status(pid) == :paused
      step = Integration.get_step(pid)
      assert step.i_step > 0

      Process.sleep(10)
      step2 = Integration.get_step(pid)
      # The i_step should not change if the simulation is in fact paused:
      assert step2.i_step == step.i_step

      Integration.continue(pid)
      assert Integration.get_status(pid) == :running

      Process.sleep(10)
      step3 = Integration.get_step(pid)
      # i_step should be incrementing again if we are running again:
      assert step3.i_step > step2.i_step

      # Change from Process.sleep() to something that periodicaly checks whether the integration is done
      Process.sleep(1_000)

      assert Integration.get_status(pid) == :completed

      {output_t, output_x} = DataSink.get_data(data_pid, pid) |> Point.split_points_into_t_and_x()

      assert length(output_t) == 201
      assert length(output_x) == 201

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end
  end
end
