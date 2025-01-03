defmodule Integrator.RungeKuttaStep do
  @moduledoc """
  The results of the computation of an individual Runge-Kutta step
  """

  @derive {Nx.Container,
   containers: [
     :t_old,
     :t_new,
     #
     :x_old,
     :x_new,
     #
     :dt,
     #
     :k_vals,
     :options_comp,
     :error_estimate
   ]}

  @type t :: %__MODULE__{
          t_old: Nx.t(),
          t_new: Nx.t(),
          #
          x_old: Nx.t(),
          x_new: Nx.t(),
          #
          dt: Nx.t(),
          #
          k_vals: Nx.t(),
          options_comp: Nx.t(),
          error_estimate: Nx.t()
        }

  defstruct [
    :t_old,
    :t_new,
    #
    :x_old,
    :x_new,
    #
    :dt,
    #
    :k_vals,
    :options_comp,
    :error_estimate
  ]
end
