defmodule Integrator.RungeKutta do
  @moduledoc """
  See the [list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
  """

  # @callback integrate(fn(), t, x, dt, k_vals) :: {}
  @callback integrate(fun(), Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: {Nx.t(), Nx.t(), Nx.t(), Nx.t()}
  # returns:     {t_next, x_next, x_est, k_new}

  # interpolate(t, x, der, t_out)
  @callback interpolate(Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()

  @callback order() :: integer()
end
