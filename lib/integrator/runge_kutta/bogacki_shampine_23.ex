defmodule Integrator.RungeKutta.BogackiShampine23 do
  @moduledoc """
  Bogacki-Shampine method of third order.  For the definition of this method see
  [Wikipedia](http://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)

  Originally based on Octave [`runge_kutta_23.m`](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_23.m)
  """
  alias Integrator.{RungeKutta, Utils}
  @behaviour RungeKutta

  import Nx.Defn

  @doc """
  Returns the order of this Runge-Kutta method (which is 3)
  """
  @impl RungeKutta
  def order, do: 3

  @a_0 [0, 0, 0]
  @a_1 [1 / 2, 0, 0]
  @a_2 [0, 3 / 4, 0]

  @b [0, 1 / 2, 3 / 4, 1]

  @c [2 / 9, 1 / 3, 4 / 9]

  @c_prime [7 / 24, 1 / 4, 1 / 3, 1 / 8]

  @doc """
  Solves a set of non-stiff Ordinary Differential Equations (non-stiff ODEs) with the well-known
  explicit Bogacki-Shampine method of order 3.

  See [Wikipedia here](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Bogacki%E2%80%93Shampine)
  and [here](https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method)
  """
  @impl RungeKutta
  defn integrate(ode_fn, t, x, dt, k_vals, t_next) do
    nx_type = Nx.type(x)

    a = Nx.tensor([@a_0, @a_1, @a_2], type: nx_type)
    b = Nx.tensor(@b, type: nx_type)
    c = Nx.tensor(@c, type: nx_type)
    c_prime = Nx.tensor(@c_prime, type: nx_type)

    s = t + dt * b
    cc = dt * c
    aa = dt * a

    last_k_vals_col = k_vals[[.., 3]]
    # Turn this into a module variable? based on precision?
    zero_tolerance = 1.0e-04
    last_col_empty? = last_k_vals_col |> Nx.abs() |> Nx.sum() < zero_tolerance

    k0 = if last_col_empty?, do: ode_fn.(t, x), else: last_k_vals_col

    k1 = ode_fn.(s[1], x + k0 * aa[1][0])
    k2 = ode_fn.(s[2], x + k1 * aa[2][1])

    k_0_2 = Nx.stack([k0, k1, k2], axis: 1)

    # 3rd order approximation
    x_next = x + Nx.dot(k_0_2, cc)

    k3 = ode_fn.(t_next, x_next)
    cc_prime = dt * c_prime
    k_new = Nx.stack([k0, k1, k2, k3], axis: 1)
    x_error_est = x + Nx.dot(k_new, cc_prime)

    {x_next, x_error_est, k_new}
  end

  @doc """
  Performs a Hermite cubic interpolation when using BogackiShampine23 via
  `Utils.hermite_cubic_interpolation/4`
  """
  @impl RungeKutta
  defn interpolate(t, x, der, t_out) do
    Utils.hermite_cubic_interpolation(t, x, der, t_out)
  end
end
