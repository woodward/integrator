defmodule Integrator.RungeKutta.DormandPrince45 do
  @moduledoc """
  Integrates and interpolates a system of ODEs with a given initial condition `x` from `t`
  to `t+dt` with the Dormand-Prince method. For the definition of this method see
  [Wikipedia](http://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method).

  It uses six function evaluations to calculate fourth and fifth-order accurate solutions.
  The difference between these solutions is then taken to be the error of the (fourth-order) solution.
  This error estimate is very convenient for adaptive stepsize integration algorithms.
  The Dormand–Prince method has seven stages, but it uses only six function evaluations per step
  because it has the FSAL (First Same As Last) property: the last stage is evaluated at the same
  point as the first stage of the next step. Dormand and Prince chose the coefficients of their
  method to minimize the error of the fifth-order solution. This is the main difference with the
  [Fehlberg method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method), which
  was constructed so that the fourth-order solution has a small error.

  [Octave docs for ode45](https://docs.octave.org/interpreter/Matlab_002dcompatible-solvers.html#XREFode45)
  """

  alias Integrator.{RungeKutta, Utils}
  @behaviour RungeKutta

  import Nx.Defn

  @a Nx.tensor(
       [
         [0, 0, 0, 0, 0, 0],
         [1 / 5, 0, 0, 0, 0, 0],
         [3 / 40, 9 / 40, 0, 0, 0, 0],
         [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
         [19_372 / 6_561, -25_360 / 2187, 64_448 / 6561, -212 / 729, 0, 0],
         [9_017 / 3_168, -355 / 33, 46_732 / 5247, 49 / 176, -5103 / 18_656, 0]
       ],
       type: :f64
     )

  @b Nx.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], type: :f64)
  @c Nx.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], type: :f64)
  @c_prime Nx.tensor([5179 / 57_600, 0, 7571 / 16_695, 393 / 640, -92_097 / 339_200, 187 / 2100, 1 / 40], type: :f64)

  @doc """
  Integrates a system of ODEs with
  [Dormand-Prince]](http://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method).

  First input argument is the function describing the system of ODEs to be integrated.

  Second input parameter is the first extreme of integration interval.

  Third input argument is the initial condition of the system.

  Fourth input argument is the timestep, that is the length of the integration interval.

  Fifth input parameter is optional and describes the Runge-Kutta evaluations
  of the previous step to use in an FSAL scheme.

  IMPLEMENT THIS SIXTH INPUT PARAMETER:
  Sixth input parameter is optional and is the time (@var{t_next}) to
  integrate to.  The default is @code{@var{t} + @var{dt}}.

  First output argument is the final integration time value.

  Second output parameter is the higher order computed solution at time `t_next` (local extrapolation).

  Third output parameter is a lower order solution for the estimation of the error.

  Fourth output parameter is matrix containing the Runge-Kutta evaluations to use in
  an FSAL (first-same-as-last) scheme or for dense output.

  Reference: Hairer, Ernst; Nørsett, Syvert Paul; Wanner, Gerhard (2008),
  Solving ordinary differential equations I: Nonstiff problems,
  Berlin, New York: Springer-Verlag, ISBN 978-3-540-56670-0
  """
  @impl RungeKutta
  defn integrate(ode_fn, t, x, dt, k_vals) do
    t_next = t + dt
    s = t + dt * @b
    cc = dt * @c
    aa = dt * @a

    slice = fn aa, row ->
      Nx.slice_along_axis(aa, row, 1) |> Nx.flatten() |> Nx.slice_along_axis(0, row)
    end

    aa_1 = slice.(aa, 1)
    # Note that aa_1 is the same as aa[1][0]
    aa_2 = slice.(aa, 2)
    aa_3 = slice.(aa, 3)
    aa_4 = slice.(aa, 4)
    aa_5 = slice.(aa, 5)

    {length_of_x} = Nx.shape(x)
    last_k_vals_col = Nx.slice_along_axis(k_vals, 6, length_of_x - 1, axis: 1) |> Nx.flatten()
    zero_tolerance = 1.0e-04
    last_col_empty? = last_k_vals_col |> Nx.abs() |> Nx.sum() < zero_tolerance

    k0 = if last_col_empty?, do: ode_fn.(t, x), else: last_k_vals_col

    k1 = ode_fn.(s[1], x + k0 * aa_1)

    k_0_1 = Nx.stack([k0, k1]) |> Nx.transpose()
    k2 = ode_fn.(s[2], x + Nx.dot(k_0_1, Nx.transpose(aa_2)))

    k_0_2 = Nx.stack([k0, k1, k2]) |> Nx.transpose()
    k3 = ode_fn.(s[3], x + Nx.dot(k_0_2, Nx.transpose(aa_3)))

    k_0_3 = Nx.stack([k0, k1, k2, k3]) |> Nx.transpose()
    k4 = ode_fn.(s[4], x + Nx.dot(k_0_3, Nx.transpose(aa_4)))

    k_0_4 = Nx.stack([k0, k1, k2, k3, k4]) |> Nx.transpose()
    k5 = ode_fn.(s[5], x + Nx.dot(k_0_4, Nx.transpose(aa_5)))

    k_0_5 = Nx.stack([k0, k1, k2, k3, k4, k5]) |> Nx.transpose()
    x_next = x + Nx.dot(k_0_5, cc)

    k6 = ode_fn.(t_next, x_next)
    k_new = Nx.stack([k0, k1, k2, k3, k4, k5, k6]) |> Nx.transpose()
    cc_prime = dt * @c_prime
    x_error_est = x + Nx.dot(k_new, cc_prime)

    {t_next, x_next, x_error_est, k_new}
  end

  @doc """
  Performs a 4th order Hermite interpolation when interpolating with DormandPrince45.
  """
  @impl RungeKutta
  defn interpolate(t, x, der, t_out) do
    Utils.hermite_quartic_interpolation(t, x, der, t_out)
  end
end