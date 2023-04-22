defmodule Integrator.RungeKutta.BogackiShampine23 do
  @moduledoc """
  Bogacki-Shampine
  """
  alias Integrator.{RungeKutta, Utils}
  @behaviour RungeKutta

  import Nx.Defn

  @impl RungeKutta
  def order, do: 3

  @impl RungeKutta
  def default_opts, do: []

  @a Nx.tensor(
       [
         [0, 0, 0],
         [1 / 2, 0, 0],
         [0, 3 / 4, 0]
       ],
       type: :f64
     )

  @b Nx.tensor([0, 1 / 2, 3 / 4, 1], type: :f64)
  @c Nx.tensor([2 / 9, 1 / 3, 4 / 9], type: :f64)
  @c_prime Nx.tensor([7 / 24, 1 / 4, 1 / 3, 1 / 8], type: :f64)

  @impl RungeKutta
  defn integrate(ode_fn, t, x, dt, k_vals) do
    s = t + dt * @b
    cc = dt * @c
    aa = dt * @a
    {length_of_x} = Nx.shape(x)
    # k = zeros (rows (x), 4);
    # k = Nx.broadcast(0.0, {length_of_x, 4})

    zero_tolerance = 1.0e-04
    last_k_vals_col = Nx.slice_along_axis(k_vals, 3, length_of_x - 1, axis: 1) |> Nx.flatten()
    last_col_empty? = last_k_vals_col |> Nx.abs() |> Nx.sum() < zero_tolerance

    k0 = if last_col_empty?, do: ode_fn.(t, x), else: last_k_vals_col

    k1 = ode_fn.(s[1], x + k0 * aa[1][0])
    k2 = ode_fn.(s[2], x + k1 * aa[2][1])

    t_next = t + dt

    k_0_2 = Nx.stack([k0, k1, k2]) |> Nx.transpose()

    # 3rd order approximation
    x_next = x + Nx.dot(k_0_2, cc)

    k3 = ode_fn.(t_next, x_next)
    cc_prime = dt * @c_prime
    k_new = Nx.stack([k0, k1, k2, k3]) |> Nx.transpose()
    x_error_est = x + Nx.dot(k_new, cc_prime)

    {t_next, x_next, x_error_est, k_new}
  end

  @impl RungeKutta
  defn interpolate(t, x, der, t_out) do
    Utils.hermite_cubic_interpolation(t, x, der, t_out)
  end
end
