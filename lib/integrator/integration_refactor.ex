defmodule Integrator.IntegrationRefactor do
  @moduledoc """
  This will become the struct in AdaptiveStepsize once it's refactored
  """

  alias Integrator.Step

  @derive {Nx.Container,
           containers: [
             :step_new,
             :step_old
           ],
           keep: []}

  @type t :: %__MODULE__{
          step_new: Step.t(),
          step_old: Step.t()
        }

  defstruct [
    :step_new,
    :step_old
  ]
end
