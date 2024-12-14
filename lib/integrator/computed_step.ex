defmodule Integrator.ComputedStep do
  @moduledoc """
  The results of the computation of an individual Runge-Kutta step
  """

  @derive {Nx.Container,
           keep: [],
           containers: [
             :t_new,
             :x_new,
             :k_vals,
             :options_comp,
             :error_estimate
           ]}

  @type t :: %__MODULE__{
          t_new: Nx.t(),
          x_new: Nx.t(),
          k_vals: Nx.t(),
          options_comp: Nx.t(),
          error_estimate: Nx.t()
        }

  defstruct [
    :t_new,
    :x_new,
    :k_vals,
    :options_comp,
    :error_estimate
  ]
end
