defmodule Integrator.Demo.BouncingBall do
  @moduledoc false

  # @spec bouncing_ball(Nx.t(), Nx.t()) :: Nx.t()
  # defn bouncing_ball(_t, x) do
  #   x0 = x[1]
  #   x1 = -9.81
  #   Nx.stack([x0, x1])
  # end

  def start_simulation(ode_fn, t_start, t_end, x0, opts) do
    do_simulate(ode_fn, Nx.to_number(t_start), t_end, x0, opts)
  end

  def simulate(_ode_fn, t_start, t_end, _x0, _opts) when t_start >= t_end do
    :halt
  end

  def simulate(ode_fn, t_start, t_end, x0, opts) do
    do_simulate(ode_fn, t_start, t_end, x0, opts)
  end

  @coefficient_of_restitution 0.9

  def do_simulate(ode_fn, t_start, t_end, x0, opts) do
    solution = Integrator.integrate(ode_fn, [t_start, t_end], x0, opts)
    new_t_start = solution.output_t |> Enum.reverse() |> hd()
    last_x = solution.output_x |> Enum.reverse() |> hd()
    new_velocity = -1.0 * @coefficient_of_restitution * Nx.to_number(last_x[1])
    new_x0 = Nx.tensor([0.0, new_velocity])
    simulate(ode_fn, Nx.to_number(new_t_start), t_end, new_x0, opts)
  end
end
