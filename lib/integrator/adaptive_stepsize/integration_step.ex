defmodule Integrator.AdaptiveStepsize.IntegrationStep do
  @moduledoc """
  A struct which represents the state of the integration as it proceeds through time
  """

  alias Integrator.RungeKutta
  import Nx.Defn

  # Values for :status_integration:
  @success 1
  @max_errors_exceeded 2
  deftransform max_errors_exceeded, do: @max_errors_exceeded

  @derive {Nx.Container,
   containers: [
     :t_current,
     :x_current,
     :dt_new,
     :rk_step,
     :fixed_output_t_next,
     :fixed_output_t_within_step?,
     #
     :count_loop__increment_step,
     :count_cycles__compute_step,
     #
     # ireject in Octave:
     :error_count,
     :i_step,
     #
     :terminal_event,
     :terminal_output,
     :status_integration,
     :status_non_linear_eqn_root,
     #
     :start_timestamp_μs,
     :step_timestamp_μs,
     :elapsed_time_μs
   ],
   keep: [
     :stepper_fn,
     :ode_fn,
     :interpolate_fn
   ]}

  @type t :: %__MODULE__{
          t_current: Nx.t(),
          x_current: Nx.t(),
          dt_new: Nx.t(),
          rk_step: RungeKutta.Step.t(),
          fixed_output_t_next: Nx.t(),
          fixed_output_t_within_step?: Nx.t(),
          #
          # interpolated_points: {Point.t(), Point.t(), Point.t(), Point.t()},
          # fixed_output_point: Point.t(),
          #
          count_loop__increment_step: Nx.t(),
          count_cycles__compute_step: Nx.t(),
          #
          # ireject in Octave:
          error_count: Nx.t(),
          i_step: Nx.t(),
          #
          terminal_event: Nx.t(),
          terminal_output: Nx.t(),
          status_integration: Nx.t(),
          status_non_linear_eqn_root: Nx.t(),
          #
          start_timestamp_μs: Nx.t(),
          step_timestamp_μs: Nx.t(),
          elapsed_time_μs: Nx.t(),
          #
          stepper_fn: fun(),
          ode_fn: fun(),
          interpolate_fn: fun()
        }
  defstruct [
    :t_current,
    :x_current,
    :dt_new,
    :rk_step,
    #
    :stepper_fn,
    :ode_fn,
    :interpolate_fn,
    #
    fixed_output_t_next: Nx.f64(0),
    fixed_output_t_within_step?: Nx.u8(0),
    #
    count_loop__increment_step: Nx.s32(0),
    count_cycles__compute_step: Nx.s32(0),
    #
    # ireject in Octave:
    error_count: Nx.s32(0),
    i_step: Nx.s32(0),
    #
    terminal_event: Nx.u8(1),
    terminal_output: Nx.s32(0),
    status_integration: Nx.u8(1),
    status_non_linear_eqn_root: Nx.u8(1),
    #
    start_timestamp_μs: Nx.s64(0),
    step_timestamp_μs: Nx.s64(0),
    elapsed_time_μs: Nx.s64(0)
  ]

  @spec new(
          stepper_fn :: RungeKutta.stepper_fn_t(),
          interpolate_fn :: RungeKutta.interpolate_fn_t(),
          ode_fn :: RungeKutta.ode_fn_t(),
          t_start :: Nx.t(),
          initial_tstep :: Nx.t(),
          x0 :: Nx.t(),
          options :: Integrator.AdaptiveStepsize.NxOptions.t(),
          start_timestamp_μs :: pos_integer()
        ) :: t()
  deftransform new(stepper_fn, interpolate_fn, ode_fn, t_start, initial_tstep, x0, options, start_timestamp_μs) do
    type = options.type

    initial_tstep = to_tensor(initial_tstep, type)
    t_start = to_tensor(t_start, type)
    x0 = to_tensor(x0, type)

    initial_tstep = Nx.min(Nx.abs(initial_tstep), options.max_step)
    initial_rk_step = RungeKutta.Step.initial_step(t_start, x0, order: options.order)

    fixed_output_t_next = Nx.add(t_start, options.fixed_output_step)

    %__MODULE__{
      t_current: t_start,
      x_current: x0,
      dt_new: initial_tstep,
      start_timestamp_μs: start_timestamp_μs,
      step_timestamp_μs: start_timestamp_μs,
      rk_step: initial_rk_step,
      fixed_output_t_next: fixed_output_t_next,
      #
      stepper_fn: stepper_fn,
      ode_fn: ode_fn,
      interpolate_fn: interpolate_fn
    }
  end

  deftransform status_integration(%__MODULE__{status_integration: status_value} = _integration_step) do
    status_integration(status_value)
  end

  deftransform status_integration(%Nx.Tensor{} = status_value) do
    status_value |> Nx.to_number() |> status_integration()
  end

  deftransform status_integration(status_value) do
    case status_value do
      @success -> :ok
      @max_errors_exceeded -> {:error, "Maximum number of errors exceeded"}
      _ -> {:error, "Unknown error"}
    end
  end

  deftransform status_non_linear_eqn_root(%__MODULE__{status_non_linear_eqn_root: status_value} = _integration_step) do
    status_non_linear_eqn_root(status_value)
  end

  deftransform status_non_linear_eqn_root(status_value) do
    Integrator.NonLinearEqnRoot.status(status_value)
  end

  @spec to_tensor(Nx.t() | float(), Nx.Type.t()) :: Nx.t()
  deftransform to_tensor(%Nx.Tensor{} = tensor, type) do
    if Nx.type(tensor) != type do
      raise "tensor #{inspect(tensor)} is of incorrect type, #{inspect(type)} expected}"
    else
      tensor
    end
  end

  deftransform to_tensor(value, type) do
    Nx.tensor(value, type: type)
  end
end
