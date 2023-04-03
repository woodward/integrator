defmodule Integrator.Utils do
  @moduledoc false

  #  DELETE LATER!!!
  #   function retval = AbsRel_norm (x, x_old, AbsTol, RelTol, normcontrol, y = zeros (size (x)))

  #   if (normcontrol)
  #     sc = max (AbsTol(:), RelTol * max (sqrt (sumsq (x)), sqrt (sumsq (x_old))));
  #     retval = sqrt (sumsq ((x - y))) / sc;
  #   else
  #     sc = max (AbsTol(:), RelTol .* max (abs (x), abs (x_old)));
  #     retval = max (abs (x - y) ./ sc);
  #   endif

  # endfunction

  def absolute_relative_norm(x, x_old, y, abs_tolerance, rel_tolerance, normcontrol: false) do
    # Octave code:
    # sc = max (AbsTol(:), RelTol .* max (abs (x), abs (x_old)));
    # retval = max (abs (x - y) ./ sc);

    sc = Nx.max(abs_tolerance, rel_tolerance |> Nx.multiply(Nx.max(Nx.abs(x), Nx.abs(x_old))))
    Nx.subtract(x, y) |> Nx.divide(sc) |> Nx.reduce_max()
  end

  def absolute_relative_norm(x, x_old, y, abs_tolerance, rel_tolerance, normcontrol: true) do
    # Octave code:
    # sc = max (AbsTol(:), RelTol * max (sqrt (sumsq (x)), sqrt (sumsq (x_old))));
    # retval = sqrt (sumsq ((x - y))) / sc;

    sumsq_x = Nx.multiply(x, x) |> Nx.sum() |> Nx.sqrt()
    sumsq_x_old = Nx.multiply(x_old, x_old) |> Nx.sum() |> Nx.sqrt()
    max_sq_x = Nx.max(sumsq_x, sumsq_x_old)
    rel_tol_x = Nx.multiply(rel_tolerance, max_sq_x)
    sc = Nx.max(abs_tolerance, rel_tol_x)

    x_minus_y = Nx.subtract(x, y)
    sum_sq_x_minus_y = Nx.multiply(x_minus_y, x_minus_y) |> Nx.sum() |> Nx.sqrt()
    Nx.divide(sum_sq_x_minus_y, sc)
  end

  def starting_stepsize() do
  end
end
