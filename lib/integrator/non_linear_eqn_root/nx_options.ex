defmodule Integrator.NonLinearEqnRoot.NxOptions do
  @moduledoc """
  `NimbleOptions` converted into an Nx-friendly `Nx.Container` struct for use when finding the non-linear eqn root
  (so that the options can be safely passed from Elixir-land to Nx-land).
  """

  alias Integrator.ExternalFnAdapter

  @derive {Nx.Container,
           containers: [
             :max_iterations,
             :max_fn_eval_count,
             :machine_eps,
             :tolerance,
             :output_fn_adapter
           ],
           keep: [
             :type
           ]}

  @type t :: %__MODULE__{
          max_iterations: Nx.t(),
          max_fn_eval_count: Nx.t(),
          type: Nx.Type.t(),
          machine_eps: Nx.t(),
          tolerance: Nx.t(),
          output_fn_adapter: ExternalFnAdapter.t()
        }

  defstruct max_iterations: 1000,
            max_fn_eval_count: 1000,
            type: {:f, 64},
            machine_eps: 0,
            tolerance: 0,
            output_fn_adapter: %ExternalFnAdapter{}
end
