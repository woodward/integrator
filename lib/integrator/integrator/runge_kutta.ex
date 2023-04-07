defmodule Integrator.RungeKutta do
  @moduledoc false

  # @callback integrate(fn(), t, x, dt, k_vals) :: {}
  @callback integrate(any(), any(), any(), any(), any()) :: {any(), any(), any(), any()}
  # returns:     {t_next, x_next, x_est, k_new}

  # @callback interpolate(t, x, der, t_out) :: {}
  @callback interpolate(any(), any(), any(), any()) :: any()

  @callback order() :: integer()
end
