defmodule Integrator.MultiIntegratorTest do
  @moduledoc false
  use Integrator.TestCase

  alias Integrator.{AdaptiveStepsize, SampleEqns, MultiIntegrator}

  describe "ballode.m" do
    setup do
      t_initial = Nx.tensor(0.0, type: :f64)
      t_final = Nx.tensor(30.0, type: :f64)
      x_initial = Nx.tensor([0.0, 20.0], type: :f64)

      opts = [
        type: :f64,
        norm_control: false,
        abs_tol: Nx.tensor(1.0e-06, type: :f64),
        rel_tol: Nx.tensor(1.0e-03, type: :f64),
        max_step: Nx.tensor(2.0, type: :f64)
      ]

      coefficient_of_restitution = Nx.tensor(-0.9, type: :f64)

      ode_fn = &SampleEqns.falling_particle/2

      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        %{status: answer, value: value}
      end

      [
        opts: opts,
        t_initial: t_initial,
        t_final: t_final,
        x_initial: x_initial,
        ode_fn: ode_fn,
        event_fn: event_fn,
        coefficient_of_restitution: coefficient_of_restitution
      ]
    end

    test "performs the integration", %{
      opts: opts,
      t_initial: t_initial,
      t_final: t_final,
      x_initial: x_initial,
      ode_fn: ode_fn,
      event_fn: event_fn,
      coefficient_of_restitution: coefficient_of_restitution
    } do
      transition_fn = fn t, x, multi, opts ->
        x0 = Nx.tensor(0.0, type: :f64)
        x1 = Nx.multiply(coefficient_of_restitution, x[1])
        x = Nx.stack([x0, x1])
        last_integration = multi.integrations |> List.first()
        first_t = last_integration.ode_t |> List.first()
        [last_t | rest_of_t] = last_integration.ode_t |> Enum.reverse()
        [next_to_last_t | _rest] = rest_of_t
        initial_step = Nx.subtract(last_t, next_to_last_t)
        max_step = Nx.subtract(last_t, first_t)
        opts = opts |> Keyword.merge(max_step: max_step, initial_step: initial_step)
        status = if length(multi.integrations) >= 10, do: :halt, else: :continue
        {status, t, x, opts}
      end

      multi = MultiIntegrator.integrate(ode_fn, event_fn, transition_fn, t_initial, t_final, x_initial, opts)

      amount_to_check = 138
      expected_t = read_nx_list("test/fixtures/octave_results/ballode/default/t.csv") |> Enum.take(amount_to_check)
      expected_x = read_nx_list("test/fixtures/octave_results/ballode/default/x.csv") |> Enum.take(amount_to_check)

      output_t = MultiIntegrator.all_output_data(multi, :output_t) |> Enum.take(amount_to_check)
      output_x = MultiIntegrator.all_output_data(multi, :output_x) |> Enum.take(amount_to_check)

      # write_t(output_t, "test/fixtures/octave_results/ballode/default/t_elixir.csv")
      # write_x(output_x, "test/fixtures/octave_results/ballode/default/x_elixir.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-02, rtol: 1.0e-02)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-02, rtol: 1.0e-02)
    end

    test "performs the integration - high fidelity multi-bounce ballode", %{
      opts: opts,
      t_initial: t_initial,
      t_final: t_final,
      x_initial: x_initial,
      ode_fn: ode_fn,
      event_fn: event_fn,
      coefficient_of_restitution: coefficient_of_restitution
    } do
      opts =
        opts
        |> Keyword.merge(
          abs_tol: Nx.tensor(1.0e-14, type: :f64),
          rel_tol: Nx.tensor(1.0e-14, type: :f64),
          norm_control: false
        )

      transition_fn = fn t, x, multi, opts ->
        x0 = Nx.tensor(0.0, type: :f64)
        x1 = Nx.multiply(coefficient_of_restitution, x[1])
        x = Nx.stack([x0, x1])
        last_integration = multi.integrations |> List.first()
        first_t = last_integration.ode_t |> List.first()
        [last_t | rest_of_t] = last_integration.ode_t |> Enum.reverse()
        [next_to_last_t | _rest] = rest_of_t
        initial_step = Nx.subtract(last_t, next_to_last_t)
        max_step = Nx.subtract(last_t, first_t)
        opts = opts |> Keyword.merge(max_step: max_step, initial_step: initial_step)
        status = if length(multi.integrations) >= 10, do: :halt, else: :continue
        {status, t, x, opts}
      end

      multi = MultiIntegrator.integrate(ode_fn, event_fn, transition_fn, t_initial, t_final, x_initial, opts)

      # Note that 153 is all of the data:
      amount_to_check = 153
      expected_t = read_nx_list("test/fixtures/octave_results/ballode/high_fidelity/t.csv") |> Enum.take(amount_to_check)
      expected_x = read_nx_list("test/fixtures/octave_results/ballode/high_fidelity/x.csv") |> Enum.take(amount_to_check)

      output_t = MultiIntegrator.all_output_data(multi, :output_t) |> Enum.take(amount_to_check)
      output_x = MultiIntegrator.all_output_data(multi, :output_x) |> Enum.take(amount_to_check)

      {t_row_72, _rest} = output_t |> List.pop_at(72)
      {x_row_72, _rest} = output_x |> List.pop_at(72)

      # Compare against Octave results:
      assert_in_delta(Nx.to_number(t_row_72), 4.07747196738022, 1.0e-17)
      assert_in_delta(Nx.to_number(x_row_72[0]), 0.0, 1.0e-13)
      assert_in_delta(Nx.to_number(x_row_72[1]), -20.0, 1.0e-13)

      # Values after first RK integration of after the first bounce:
      {t_row_76, _rest} = output_t |> List.pop_at(76)
      {x_row_76, _rest} = output_x |> List.pop_at(76)

      # Compare against Octave results:
      assert_in_delta(Nx.to_number(t_row_76), 5.256295464839447, 1.0e-14)
      assert_in_delta(Nx.to_number(x_row_76[0]), 14.40271312308144, 1.0e-13)
      assert_in_delta(Nx.to_number(x_row_76[1]), 6.435741489925020, 1.0e-14)

      # write_t(output_t, "test/fixtures/octave_results/ballode/high_fidelity/t_elixir.csv")
      # write_x(output_x, "test/fixtures/octave_results/ballode/high_fidelity/x_elixir.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-07, rtol: 1.0e-07)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-07, rtol: 1.0e-07)

      t_last_row = output_t |> List.last()
      x_last_row = output_x |> List.last()

      # Compare against Octave results:
      assert_in_delta(Nx.to_number(t_last_row), 26.55745402242623, 1.0e-14)
      assert_in_delta(Nx.to_number(x_last_row[0]), -1.360023205165817e-13, 1.0e-12)
      assert_in_delta(Nx.to_number(x_last_row[1]), -7.748409780000432, 1.0e-12)
    end

    test "can terminate the simulation based on some event (in this case 2 bounces)", %{
      opts: opts,
      t_initial: t_initial,
      t_final: t_final,
      x_initial: x_initial,
      ode_fn: ode_fn,
      event_fn: event_fn,
      coefficient_of_restitution: coefficient_of_restitution
    } do
      number_of_bounces = 2

      transition_fn = fn t, x, multi, opts ->
        x0 = Nx.tensor(0.0, type: :f64)
        x1 = Nx.multiply(coefficient_of_restitution, x[1])
        x = Nx.stack([x0, x1])
        last_integration = multi.integrations |> List.first()
        first_t = last_integration.ode_t |> List.first()
        [last_t | rest_of_t] = last_integration.ode_t |> Enum.reverse()
        [next_to_last_t | _rest] = rest_of_t
        initial_step = Nx.subtract(last_t, next_to_last_t)
        max_step = Nx.subtract(last_t, first_t)
        opts = opts |> Keyword.merge(max_step: max_step, initial_step: initial_step)
        status = if length(multi.integrations) >= number_of_bounces, do: :halt, else: :continue
        {status, t, x, opts}
      end

      multi = MultiIntegrator.integrate(ode_fn, event_fn, transition_fn, t_initial, t_final, x_initial, opts)

      amount_to_check = 53
      expected_t = read_nx_list("test/fixtures/octave_results/ballode/default/t.csv") |> Enum.take(amount_to_check)
      expected_x = read_nx_list("test/fixtures/octave_results/ballode/default/x.csv") |> Enum.take(amount_to_check)

      output_t = MultiIntegrator.all_output_data(multi, :output_t)
      output_x = MultiIntegrator.all_output_data(multi, :output_x)

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-02, rtol: 1.0e-02)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-02, rtol: 1.0e-02)
    end
  end

  describe "all_output_data/2" do
    test "gets the output_t values from all of the simuations" do
      sim1 = %AdaptiveStepsize{output_t: [1, 2, 3]}
      sim2 = %AdaptiveStepsize{output_t: [3, 4, 5]}
      sim3 = %AdaptiveStepsize{output_t: [5, 6, 7]}
      multi = %MultiIntegrator{integrations: [sim1, sim2, sim3]}

      output_t = MultiIntegrator.all_output_data(multi, :output_t)

      assert output_t == [1, 2, 3, 4, 5, 6, 7]
    end

    test "gets the output_x values from all of the simuations" do
      sim1 = %AdaptiveStepsize{output_x: [1, 2, 3]}
      sim2 = %AdaptiveStepsize{output_x: [3, 4, 5]}
      sim3 = %AdaptiveStepsize{output_x: [5, 6, 7]}
      multi = %MultiIntegrator{integrations: [sim1, sim2, sim3]}

      output_t = MultiIntegrator.all_output_data(multi, :output_x)

      assert output_t == [1, 2, 3, 4, 5, 6, 7]
    end
  end
end
