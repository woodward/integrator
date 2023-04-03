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

  def absolute_relative_norm(x, x_old, y, absolute_tolerance, relative_tolerance, normcontrol: false) do
    # Octave code:
    # sc = max (AbsTol(:), RelTol .* max (abs (x), abs (x_old)));
    # retval = max (abs (x - y) ./ sc);

    sc = Nx.max(absolute_tolerance, relative_tolerance |> Nx.multiply(Nx.max(Nx.abs(x), Nx.abs(x_old))))
    Nx.subtract(x, y) |> Nx.divide(sc) |> Nx.reduce_max()
  end

  def starting_stepsize() do
  end
end
