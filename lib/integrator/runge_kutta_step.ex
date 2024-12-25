defmodule Integrator.RungeKuttaStep do
  @moduledoc """
  The results of the computation of an individual Runge-Kutta step
  """

  @derive {Nx.Container,
           containers: [
             :t,
             :x,
             :k_vals,
             :options_comp,
             :error_estimate
           ],
           keep: []}

  @type t :: %__MODULE__{
          t: Nx.t(),
          x: Nx.t(),
          k_vals: Nx.t(),
          options_comp: Nx.t(),
          error_estimate: Nx.t()
        }

  defstruct [
    :t,
    :x,
    :k_vals,
    :options_comp,
    :error_estimate
  ]
end
