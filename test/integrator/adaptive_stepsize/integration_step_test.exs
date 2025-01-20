defmodule Integrator.AdaptiveStepsize.IntegrationStepTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsizeRefactor
  alias Integrator.RungeKutta.BogackiShampine23
  alias Integrator.RungeKutta.DormandPrince45
  alias Integrator.SampleEqns

  import Nx, only: :sigils

  describe "new" do
    test "returns an initial step for a Dormand-Prince45 integration" do
      stepper_fn = &DormandPrince45.integrate/6
      interpolate_fn = &DormandPrince45.interpolate/4
      ode_fn = &SampleEqns.van_der_pol_fn/2
      start_timestamp_μs = 9

      t_start = Nx.f64(0.0)
      x0 = Nx.f64([2.0, 0.0])
      initial_tstep = Nx.f64(0.068129)

      options = initial_nx_options_dormand_prince()

      step = IntegrationStep.new(stepper_fn, interpolate_fn, ode_fn, t_start, initial_tstep, x0, options, start_timestamp_μs)

      expected_initial_k_vals = ~MAT[
              0.0  0.0  0.0  0.0  0.0  0.0  0.0
              0.0  0.0  0.0  0.0  0.0  0.0  0.0
            ]f64

      nan = Nx.Constants.nan(:f64)

      expected_step = %Integrator.AdaptiveStepsize.IntegrationStep{
        t_current: Nx.f64(0.0),
        x_current: Nx.f64([2.0, 0.0]),
        dt_new: Nx.f64(0.068129),
        rk_step: %Integrator.RungeKutta.Step{
          t_old: nan,
          t_new: Nx.f64(0.0),
          x_old: Nx.f64([0.0, 0.0]),
          x_new: Nx.f64([2.0, 0.0]),
          k_vals: expected_initial_k_vals,
          options_comp: Nx.f64(0.0),
          error_estimate: nan,
          dt: nan
        },
        stepper_fn: &Integrator.RungeKutta.DormandPrince45.integrate/6,
        ode_fn: &Integrator.SampleEqns.van_der_pol_fn/2,
        interpolate_fn: &Integrator.RungeKutta.DormandPrince45.interpolate/4,
        fixed_output_t_next: Nx.f64(0.0),
        fixed_output_t_within_step?: Nx.u8(0),
        count_loop__increment_step: Nx.s32(0),
        count_cycles__compute_step: Nx.s32(0),
        error_count: Nx.s32(0),
        i_step: Nx.s32(0),
        terminal_event: Nx.u8(1),
        terminal_output: Nx.s32(0),
        status_integration: Nx.u8(1),
        status_non_linear_eqn_root: Nx.u8(0),
        start_timestamp_μs: start_timestamp_μs,
        step_timestamp_μs: start_timestamp_μs,
        elapsed_time_μs: Nx.s64(0)
      }

      assert step == expected_step
    end

    test "returns an initial step for a Bogacki-Shampine23 integration" do
      stepper_fn = &BogackiShampine23.integrate/6
      interpolate_fn = &BogackiShampine23.interpolate/4
      ode_fn = &SampleEqns.van_der_pol_fn/2
      start_timestamp_μs = 9

      t_start = Nx.f64(0.0)
      x0 = Nx.f64([2.0, 0.0])
      initial_tstep = Nx.f64(0.068129)

      options = initial_nx_options_bobacki_shampine()

      step = IntegrationStep.new(stepper_fn, interpolate_fn, ode_fn, t_start, initial_tstep, x0, options, start_timestamp_μs)

      expected_initial_k_vals = ~MAT[
              0.0  0.0  0.0  0.0
              0.0  0.0  0.0  0.0
            ]f64

      nan = Nx.Constants.nan(:f64)

      expected_step = %Integrator.AdaptiveStepsize.IntegrationStep{
        t_current: Nx.f64(0.0),
        x_current: Nx.f64([2.0, 0.0]),
        dt_new: Nx.f64(0.068129),
        rk_step: %Integrator.RungeKutta.Step{
          t_old: nan,
          t_new: Nx.f64(0.0),
          x_old: Nx.f64([0.0, 0.0]),
          x_new: Nx.f64([2.0, 0.0]),
          k_vals: expected_initial_k_vals,
          options_comp: Nx.f64(0.0),
          error_estimate: nan,
          dt: nan
        },
        stepper_fn: &Integrator.RungeKutta.BogackiShampine23.integrate/6,
        ode_fn: &Integrator.SampleEqns.van_der_pol_fn/2,
        interpolate_fn: &Integrator.RungeKutta.BogackiShampine23.interpolate/4,
        fixed_output_t_next: Nx.f64(0.0),
        fixed_output_t_within_step?: Nx.u8(0),
        count_loop__increment_step: Nx.s32(0),
        count_cycles__compute_step: Nx.s32(0),
        error_count: Nx.s32(0),
        i_step: Nx.s32(0),
        terminal_event: Nx.u8(1),
        terminal_output: Nx.s32(0),
        status_integration: Nx.u8(1),
        status_non_linear_eqn_root: Nx.u8(0),
        start_timestamp_μs: start_timestamp_μs,
        step_timestamp_μs: start_timestamp_μs,
        elapsed_time_μs: Nx.s64(0)
      }

      assert step == expected_step
    end
  end

  describe "status_integration/1" do
    test "returns :ok if the integration was successful" do
      integration = %IntegrationStep{status_integration: Nx.u8(1)}
      assert IntegrationStep.status_integration(integration) == :ok

      integration = %IntegrationStep{status_integration: Nx.s32(1)}
      assert IntegrationStep.status_integration(integration) == :ok

      integration = %IntegrationStep{status_integration: 1}
      assert IntegrationStep.status_integration(integration) == :ok
    end

    test "returns an error tuple for max errors exceeded" do
      integration = %IntegrationStep{status_integration: IntegrationStep.max_errors_exceeded()}
      assert IntegrationStep.status_integration(integration) == {:error, "Maximum number of errors exceeded"}
    end

    test "returns an error tuple for an unknown error" do
      integration = %IntegrationStep{status_integration: 999}
      assert IntegrationStep.status_integration(integration) == {:error, "Unknown error"}
    end
  end

  describe "status_non_linear_eqn_root/1" do
    test "calls through to the NonLinearEqnRoot functions to get the status" do
      integration = %IntegrationStep{status_non_linear_eqn_root: Nx.u8(2)}
      assert IntegrationStep.status_non_linear_eqn_root(integration) == {:error, "Invalid initial bracket"}
    end
  end

  defp initial_nx_options_dormand_prince do
    %AdaptiveStepsizeRefactor.NxOptions{
      abs_tol: Nx.f64(1.0e-06),
      rel_tol: Nx.f64(1.0e-03),
      norm_control?: Nx.u8(0),
      order: 5,
      fixed_output_times?: Nx.u8(0),
      fixed_output_step: Nx.f64(0.0),
      speed: Nx.Constants.infinity(:f64),
      refine: 4,
      type: {:f, 64},
      max_step: Nx.f64(2.0),
      max_number_of_errors: Nx.s32(5000),
      nx_while_loop_integration?: Nx.u8(1),
      event_fn_adapter: %Integrator.ExternalFnAdapter{
        external_fn: &Integrator.ExternalFnAdapter.no_op_double_arity_fn/2
      },
      output_fn_adapter: %Integrator.ExternalFnAdapter{
        external_fn: &Integrator.ExternalFnAdapter.no_op_fn/1
      },
      zero_fn_adapter: %Integrator.ExternalFnAdapter{
        external_fn: &Integrator.ExternalFnAdapter.no_op_fn/1
      },
      non_linear_eqn_root_nx_options: %Integrator.NonLinearEqnRoot.NxOptions{
        max_iterations: 1000,
        max_fn_eval_count: 1000,
        type: {:f, 64},
        machine_eps: Nx.Constants.epsilon(:f64),
        tolerance: Nx.Constants.epsilon(:f64),
        output_fn_adapter: %Integrator.ExternalFnAdapter{
          external_fn: &Integrator.ExternalFnAdapter.no_op_fn/1
        }
      }
    }
  end

  defp initial_nx_options_bobacki_shampine do
    %AdaptiveStepsizeRefactor.NxOptions{
      abs_tol: Nx.f64(1.0e-06),
      rel_tol: Nx.f64(1.0e-03),
      norm_control?: Nx.u8(0),
      order: 3,
      fixed_output_times?: Nx.u8(0),
      fixed_output_step: Nx.f64(0.0),
      speed: Nx.Constants.infinity(:f64),
      refine: 4,
      type: {:f, 64},
      max_step: Nx.f64(2.0),
      max_number_of_errors: Nx.s32(5000),
      nx_while_loop_integration?: Nx.u8(1),
      event_fn_adapter: %Integrator.ExternalFnAdapter{
        external_fn: &Integrator.ExternalFnAdapter.no_op_double_arity_fn/2
      },
      output_fn_adapter: %Integrator.ExternalFnAdapter{
        external_fn: &Integrator.ExternalFnAdapter.no_op_fn/1
      },
      zero_fn_adapter: %Integrator.ExternalFnAdapter{
        external_fn: &Integrator.ExternalFnAdapter.no_op_fn/1
      },
      non_linear_eqn_root_nx_options: %Integrator.NonLinearEqnRoot.NxOptions{
        max_iterations: 1000,
        max_fn_eval_count: 1000,
        type: {:f, 64},
        machine_eps: Nx.Constants.epsilon(:f64),
        tolerance: Nx.Constants.epsilon(:f64),
        output_fn_adapter: %Integrator.ExternalFnAdapter{
          external_fn: &Integrator.ExternalFnAdapter.no_op_fn/1
        }
      }
    }
  end
end
