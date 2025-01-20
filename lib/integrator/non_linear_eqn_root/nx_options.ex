defmodule Integrator.NonLinearEqnRoot.NxOptions do
  @moduledoc """
  `NimbleOptions` converted into an Nx-friendly `Nx.Container` struct for use when finding the non-linear eqn root
  (so that the options can be safely passed from Elixir-land to Nx-land).
  """

  alias Integrator.ExternalFnAdapter
  alias Integrator.NonLinearEqnRoot

  import Nx.Defn

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

  @spec convert_opts_to_nx_options(Keyword.t()) :: t()
  deftransform convert_opts_to_nx_options(opts) do
    nimble_opts = opts |> NimbleOptions.validate!(NonLinearEqnRoot.options_schema()) |> Map.new()
    nx_type = nimble_opts[:type] |> Nx.Type.normalize!()
    machine_eps = default_to_epsilon(nimble_opts[:machine_eps], nx_type)
    tolerance = default_to_epsilon(nimble_opts[:tolerance], nx_type)
    output_fn_adapter = ExternalFnAdapter.wrap_external_fn(nimble_opts[:nonlinear_eqn_root_output_fn])

    %__MODULE__{
      machine_eps: machine_eps,
      max_fn_eval_count: nimble_opts[:max_fn_eval_count],
      max_iterations: nimble_opts[:max_iterations],
      output_fn_adapter: output_fn_adapter,
      tolerance: tolerance,
      type: nx_type
    }
  end

  deftransformp default_to_epsilon(nil, type), do: Nx.Constants.epsilon(type)
  deftransformp default_to_epsilon(value, type), do: Nx.tensor(value, type: type)
end
