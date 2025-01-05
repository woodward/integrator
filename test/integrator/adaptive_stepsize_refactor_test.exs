defmodule Integrator.AdaptiveStepsizeRefactorTest do
  @moduledoc false
  use Integrator.TestCase, async: true
  import Nx, only: :sigils

  alias Integrator.AdaptiveStepsizeRefactor
  alias Integrator.SampleEqns
  alias Integrator.RungeKutta.DormandPrince45

  describe "integrate" do
    @tag :skip
    test "works - no data interpolation (refine == 1)" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      order = DormandPrince45.order()
      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

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
        max_step: Nx.tensor(2.0, type: :f64),
        output_fn: output_fn
      ]

      # From Octave (or equivalently, from AdaptiveStepsize.starting_stepsize/7):
      initial_tstep = Nx.tensor(0.068129, type: :f64)

      result =
        AdaptiveStepsizeRefactor.integrate(
          stepper_fn,
          interpolate_fn,
          ode_fn,
          t_start,
          t_end,
          nil,
          initial_tstep,
          x0,
          order,
          opts
        )

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

      assert result.overall_elapsed_time_μs(result) > 1
      assert result.step_elapsed_time_μs(result) > 1
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
end
