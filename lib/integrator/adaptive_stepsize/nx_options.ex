defmodule Integrator.AdaptiveStepsize.NxOptions do
  @moduledoc """
  `NimbleOptions` converted into an Nx-friendly `Nx.Container` struct for use when finding the non-linear eqn root
  (so that the options can be safely passed from Elixir-land to Nx-land).
  """

  alias Integrator.ExternalFnAdapter
  alias Integrator.NonLinearEqnRoot

  @derive {Nx.Container,
   containers: [
     :abs_tol,
     :rel_tol,
     :norm_control?,
     :fixed_output_times?,
     :fixed_output_step,
     :speed,
     :max_step,
     :max_number_of_errors,
     :nx_while_loop_integration?,
     #
     :event_fn_adapter,
     :output_fn_adapter,
     :zero_fn_adapter,
     #
     :non_linear_eqn_root_nx_options
   ],
   keep: [
     :order,
     :refine,
     :type
   ]}

  @type t :: %__MODULE__{
          abs_tol: Nx.t(),
          rel_tol: Nx.Type.t(),
          norm_control?: Nx.t(),
          order: integer(),
          fixed_output_times?: Nx.t(),
          fixed_output_step: Nx.t(),
          speed: Nx.t(),
          refine: integer(),
          type: Nx.Type.t(),
          max_step: Nx.t(),
          max_number_of_errors: Nx.t(),
          nx_while_loop_integration?: Nx.t(),
          #
          event_fn_adapter: ExternalFnAdapter.t(),
          output_fn_adapter: ExternalFnAdapter.t(),
          zero_fn_adapter: ExternalFnAdapter.t(),
          #
          non_linear_eqn_root_nx_options: NonLinearEqnRoot.NxOptions.t()
        }

  # The default values here are just placeholders; the actual defaults come from NimbleOpts
  # (and are then converted to Nx in convert_to_nx_options/3)
  defstruct abs_tol: 1000.0,
            rel_tol: 1000.0,
            norm_control?: Nx.u8(1),
            order: 0,
            fixed_output_times?: Nx.u8(0),
            fixed_output_step: 1000.0,
            speed: Nx.Constants.nan(:f64),
            refine: 0,
            type: {:f, 64},
            max_step: 0.0,
            max_number_of_errors: 0,
            nx_while_loop_integration?: 1,
            #
            event_fn_adapter: %ExternalFnAdapter{},
            output_fn_adapter: %ExternalFnAdapter{},
            zero_fn_adapter: %ExternalFnAdapter{},
            #
            non_linear_eqn_root_nx_options: %NonLinearEqnRoot.NxOptions{}
end
