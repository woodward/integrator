defmodule Integrator.AdaptiveStepsize.IntegrationStep do
  @moduledoc """
  Integrates a set of ODEs with an adaptive timestep.
  """

  alias Integrator.Point
  alias Integrator.RungeKutta

  @derive {Nx.Container,
   containers: [
     :t_at_start_of_step,
     :x_at_start_of_step,
     :dt_new,
     :rk_step,
     :fixed_output_t_next,
     :fixed_output_t_within_step?,
     # perhaps status is not necessary, and terminal_event is used intead?
     :status,
     #
     # Perhaps none of these three are needed if I push out the points out immediately?
     :output_point,
     :interpolated_points,
     :fixed_output_point,
     :output_t_and_x,
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
     #
     :step_start_timestamp_μs,
     :step_elapsed_time_μs,
     #
     :overall_start_timestamp_μs,
     :overall_elapsed_time_μs
   ],
   keep: [
     :stepper_fn,
     :ode_fn,
     :interpolate_fn
   ]}

  @type t :: %__MODULE__{
          t_at_start_of_step: Nx.t(),
          x_at_start_of_step: Nx.t(),
          dt_new: Nx.t(),
          rk_step: RungeKutta.Step.t(),
          fixed_output_t_next: Nx.t(),
          fixed_output_t_within_step?: Nx.t(),
          # perhaps status is not necessary, and terminal_event is used intead?
          status: Nx.t(),
          #
          output_point: Point.t(),
          interpolated_points: {},
          fixed_output_point: {},
          output_t_and_x: {Nx.t(), Nx.t()},
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
          #
          step_start_timestamp_μs: Nx.t(),
          step_elapsed_time_μs: Nx.t(),
          #
          overall_start_timestamp_μs: Nx.t(),
          overall_elapsed_time_μs: Nx.t(),
          #
          stepper_fn: fun(),
          ode_fn: fun(),
          interpolate_fn: fun()
        }
  defstruct [
    :t_at_start_of_step,
    :x_at_start_of_step,
    :dt_new,
    :rk_step,
    #
    :stepper_fn,
    :ode_fn,
    :interpolate_fn,
    #
    fixed_output_t_next: Nx.f64(0),
    fixed_output_t_within_step?: Nx.u8(0),
    # perhaps status is not necessary, and terminal_event is used intead?
    status: Nx.u8(1),
    #
    output_point: %Point{},
    interpolated_points: {},
    fixed_output_point: {},
    output_t_and_x: {},
    #
    count_loop__increment_step: Nx.s32(0),
    count_cycles__compute_step: Nx.s32(0),
    #
    # ireject in Octave:
    error_count: Nx.s32(0),
    i_step: Nx.s32(0),
    #
    terminal_event: Nx.s32(0),
    terminal_output: Nx.s32(0),
    #
    step_start_timestamp_μs: Nx.s32(0),
    step_elapsed_time_μs: Nx.s32(0),
    #
    overall_start_timestamp_μs: Nx.s32(0),
    overall_elapsed_time_μs: Nx.s32(0)
  ]
end
