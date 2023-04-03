defmodule Integrator.Utils do
  @moduledoc false
  import Nx.Defn

  @doc """
  Based on
  [Octave function AbsRelNorm](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/AbsRel_norm.m)

  ## Options
  * `:norm_control` - Control error relative to norm; i.e., control the error `e` at each step using the norm of the
    solution rather than its absolute value.  Defaults to true.

    See [Matlab documentation](https://www.mathworks.com/help/matlab/ref/odeset.html#bu2m9z6-NormControl)
    for a description of norm control.
  """
  defn abs_rel_norm(x, x_old, y, abs_tolerance, rel_tolerance, opts \\ []) do
    opts = keyword!(opts, norm_control: true)

    if opts[:norm_control] do
      # Octave code
      # sc = max (AbsTol(:), RelTol * max (sqrt (sumsq (x)), sqrt (sumsq (x_old))));
      # retval = sqrt (sumsq ((x - y))) / sc;

      max_sq_x = Nx.max(sum_sq(x), sum_sq(x_old))
      sc = Nx.max(abs_tolerance, rel_tolerance * max_sq_x)
      sum_sq(x - y) / sc
    else
      # Octave code:
      # sc = max (AbsTol(:), RelTol .* max (abs (x), abs (x_old)));
      # retval = max (abs (x - y) ./ sc);

      sc = Nx.max(abs_tolerance, rel_tolerance * Nx.max(Nx.abs(x), Nx.abs(x_old)))
      (Nx.abs(x - y) / sc) |> Nx.reduce_max()
    end
  end

  def starting_stepsize() do
  end

  defnp sum_sq(x) do
    (x * x) |> Nx.sum() |> Nx.sqrt()
  end
end
