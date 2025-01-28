defmodule Integrator.IntegrationTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.DataCollector
  alias Integrator.Point

  describe "can start up and run a simulation" do
    setup do
      initial_x = Nx.f64([2.0, 0.0])
      t_initial = Nx.f64(0.0)
      t_final = Nx.f64(20.0)

      [initial_x: initial_x, t_initial: t_initial, t_final: t_final]
    end

    test "performs the integration", %{initial_x: initial_x, t_initial: t_initial, t_final: t_final} do
      # See:
      # https://octave.sourceforge.io/octave_results/function/ode45.html
      #
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0]);

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      opts = [
        type: :f64,
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03),
        norm_control?: false,
        max_step: Nx.f64(2.0),
        output_fn: output_fn
      ]

      _solution = Integrator.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_x, opts)

      {output_t, output_x} = DataCollector.get_data(pid) |> Point.split_points_into_t_and_x()
      assert length(output_t) == 201
      assert length(output_x) == 201

      expected_t = read_nx_list("test/fixtures/octave_results/van_der_pol/default/t.csv")
      expected_x = read_nx_list("test/fixtures/octave_results/van_der_pol/default/x.csv")

      assert_nx_lists_equal(output_t, expected_t, atol: 1.0e-04, rtol: 1.0e-04)
      assert_nx_lists_equal(output_x, expected_x, atol: 1.0e-04, rtol: 1.0e-04)
    end
  end
end
