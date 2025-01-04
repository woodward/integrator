defmodule Integrator.FixedOutputTimes do
  @moduledoc """
  Not in use yet; just exploring ideas
  """

  @derive {Nx.Container,
           containers: [
             :t_last,
             :t_next,
             :dt_fixed
           ]}

  @type t :: %__MODULE__{
          t_last: Nx.t(),
          t_next: Nx.t(),
          dt_fixed: Nx.t()
        }

  defstruct [
    :t_last,
    :t_next,
    :dt_fixed
  ]
end
