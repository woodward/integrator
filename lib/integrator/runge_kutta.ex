defmodule Integrator.RungeKutta do
  @moduledoc """
  See the [list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
  """

  @doc """
  Integrates an ODE function
  """
  # @callback integrate(fn(), t, x, dt, k_vals) :: {}
  @callback integrate(fun(), Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: {Nx.t(), Nx.t(), Nx.t(), Nx.t()}
  # returns:     {t_next, x_next, x_est, k_new}

  @doc """
  Interpolates using the method that is suitable for this particular Runge-Kutta method
  """
  # interpolate(t, x, der, t_out)
  @callback interpolate(Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()

  @doc """
  The order of this Runge-Kutta method
  """
  @callback order() :: integer()
end
