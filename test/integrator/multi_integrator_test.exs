defmodule Integrator.MultiIntegratorTest do
  @moduledoc false
  use Integrator.TestCase

  alias Integrator.{AdaptiveStepsize, MultiIntegrator}

  describe "ballode.m" do
    test "performs the integration" do
      t_initial = Nx.tensor(0.0, type: :f64)
      t_final = Nx.tensor(30.0, type: :f64)
      x_initial = Nx.tensor([0.0, 20.0], type: :f64)
      opts = [type: :f64, norm_control: false]

      ode_fn = fn _t, x ->
        x0 = x[1]
        x1 = Nx.tensor(-9.81, type: :f64)
        Nx.stack([x0, x1])
      end

      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        %{status: answer, value: value}
      end

      coefficient_of_restitution = -0.9

      transition_fn = fn t, x, multi, opts ->
        x0 = Nx.tensor(0.0, type: :f64)
        x1 = Nx.multiply(coefficient_of_restitution, x[1])
        x = Nx.stack([x0, x1])
        last_integration = multi.integrations |> List.first()
        first_t = last_integration.ode_t |> List.first()
        [last_t | rest_of_t] = last_integration.ode_t |> Enum.reverse()
        [next_to_last_t | _rest] = rest_of_t
        initial_step = Nx.to_number(last_t) - Nx.to_number(next_to_last_t)
        max_step = Nx.to_number(last_t) - Nx.to_number(first_t)
        opts = opts |> Keyword.merge(max_step: max_step, initial_step: initial_step)
        status = if length(multi.integrations) >= 10, do: :halt, else: :continue
        {status, t, x, opts}
      end

      multi = MultiIntegrator.integrate(ode_fn, event_fn, transition_fn, t_initial, t_final, x_initial, opts)

      amount_to_check = 120
      expected_t = read_nx_list("test/fixtures/octave_results/ballode/t.csv") |> Enum.take(amount_to_check)
      expected_x = read_nx_list("test/fixtures/octave_results/ballode/x.csv") |> Enum.take(amount_to_check)

      output_t = MultiIntegrator.all_output_data(multi, :output_t) |> Enum.take(amount_to_check)
      output_x = MultiIntegrator.all_output_data(multi, :output_x) |> Enum.take(amount_to_check)

      # write_t(output_t, "test/fixtures/octave_results/ballode/t_elixir.csv")
      # write_x(output_x, "test/fixtures/octave_results/ballode/x_elixir.csv")

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
