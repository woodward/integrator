defmodule Integrator.MultiIntegratorTest do
  @moduledoc false
  use Integrator.TestCase

  alias Integrator.{AdaptiveStepsize, MultiIntegrator}

  describe "ballode.m" do
    @tag :skip
    test "performs the integration" do
      t_initial = Nx.tensor(0.0, type: :f64)
      t_final = Nx.tensor(30.0, type: :f64)
      x_initial = Nx.tensor([0.0, 20.0], type: :f64)
      opts = [type: :f64]

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

      transition_fn = fn _t, x ->
        x1 = Nx.multiply(coefficient_of_restitution, x[1])
        Nx.stack([x[0], x1])
      end

      multi = MultiIntegrator.integrate(ode_fn, event_fn, transition_fn, t_initial, t_final, x_initial, opts)

      expected_t = read_nx_list("test/fixtures/octave_results/ballode/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/ballode/x.csv")

      output_t = MultiIntegrator.all_output_data(multi, :output_t)
      output_x = MultiIntegrator.all_output_data(multi, :output_x)

      write_t(output_t, "test/fixtures/octave_results/ballode/t_elixir.csv")
      write_x(output_x, "test/fixtures/octave_results/ballode/x_elixir.csv")

      # assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      # assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
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
