defmodule Integrator.RungeKutta.BogackiShampine23 do
  @moduledoc """
  Bogacki-Shampine method of third order.  For the definition of this method see
  [Wikipedia](http://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)

  Originally based on Octave [`runge_kutta_23.m`](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/runge_kutta_23.m)
  """
  alias Integrator.{RungeKutta, Utils}
  @behaviour RungeKutta

  import Nx.Defn

  @impl RungeKutta
  def order, do: 3

  @a_f64 Nx.tensor(
           [
             [0, 0, 0],
             [1 / 2, 0, 0],
             [0, 3 / 4, 0]
           ],
           type: :f64
         )
  @a %{f64: @a_f64, f32: Nx.as_type(@a_f64, :f32)}

  @b_f64 Nx.tensor([0, 1 / 2, 3 / 4, 1], type: :f64)
  @b %{f64: @b_f64, f32: Nx.as_type(@b_f64, :f32)}

  @c_f64 Nx.tensor([2 / 9, 1 / 3, 4 / 9], type: :f64)
  @c %{f64: @c_f64, f32: Nx.as_type(@c_f64, :f32)}

  @c_prime_f64 Nx.tensor([7 / 24, 1 / 4, 1 / 3, 1 / 8], type: :f64)
  @c_prime %{f64: @c_prime_f64, f32: Nx.as_type(@c_prime_f64, :f32)}

  @doc """
  Solves a set of non-stiff Ordinary Differential Equations (non-stiff ODEs) with the well-known
  explicit Bogacki-Shampine method of order 3.

  See [Wikipedia here](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Bogacki%E2%80%93Shampine)
  and [here](https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method)
  """
  @impl RungeKutta
  defn integrate(ode_fn, t, x, dt, k_vals) do
    type = :f64
    # type = type_atom(x)

    s = t + dt * @b[type]
    cc = dt * @c[type]
    aa = dt * @a[type]
    # k = zeros (rows (x), 4);
    # k = Nx.broadcast(0.0, {length_of_x, 4})

    zero_tolerance = 1.0e-04
    last_k_vals_col = Nx.slice_along_axis(k_vals, 3, 1, axis: 1) |> Nx.flatten()
    last_col_empty? = last_k_vals_col |> Nx.abs() |> Nx.sum() < zero_tolerance

    k0 = if last_col_empty?, do: ode_fn.(t, x), else: last_k_vals_col

    k1 = ode_fn.(s[1], x + k0 * aa[1][0])
    k2 = ode_fn.(s[2], x + k1 * aa[2][1])

    t_next = t + dt

    k_0_2 = Nx.stack([k0, k1, k2]) |> Nx.transpose()

    # 3rd order approximation
    x_next = x + Nx.dot(k_0_2, cc)

    k3 = ode_fn.(t_next, x_next)
    cc_prime = dt * @c_prime[type]
    k_new = Nx.stack([k0, k1, k2, k3]) |> Nx.transpose()
    x_error_est = x + Nx.dot(k_new, cc_prime)

    {t_next, x_next, x_error_est, k_new}
  end

  @doc """
  Performs a Hermite cubic interpolation when using BogackiShampine23 via
  `Utils.hermite_cubic_interpolation/4`
  """
  @impl RungeKutta
  defn interpolate(t, x, der, t_out) do
    Utils.hermite_cubic_interpolation(t, x, der, t_out)
  end

  @spec type_atom(Nx.t()) :: atom()
  deftransformp type_atom(tensor), do: Utils.type_atom(tensor)
end
