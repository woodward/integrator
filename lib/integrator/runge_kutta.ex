defmodule Integrator.RungeKutta do
  @moduledoc """
  See the [list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
  """

  @doc """
  Integrates an ODE function
  """
  @callback integrate(ode_fn :: fun(), t :: Nx.t(), x :: Nx.t(), dt :: Nx.t(), k_vals :: Nx.t()) ::
              {t_next :: Nx.t(), x_next :: Nx.t(), x_est :: Nx.t(), k_new :: Nx.t()}

  @doc """
  Interpolates using the method that is suitable for this particular Runge-Kutta method
  """
  @callback interpolate(t :: Nx.t(), x :: Nx.t(), der :: Nx.t(), t_out :: Nx.t()) :: x_out :: Nx.t()

  @doc """
  The order of this Runge-Kutta method
  """
  @callback order() :: integer()
end
