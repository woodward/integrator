defmodule Integrator.Integration do
  @moduledoc """
  A struct which maintains the state of the integration
  """

  alias Integrator.Step

  @derive {Nx.Container,
           keep: [],
           containers: [
             :step_new,
             :step_old
           ]}

  @type t :: %__MODULE__{
          step_new: Step.t(),
          step_old: Step.t()
        }

  defstruct [
    :step_new,
    :step_old
  ]
end
