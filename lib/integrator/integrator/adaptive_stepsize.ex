defmodule Integrator.AdaptiveStepsize do
  @moduledoc false
  import Nx.Defn
  alias Integrator.Utils

  defn integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, x0, order, opts \\ []) do
    Utils.zero_vector(x0)
  end

  @doc """
  Implements the Kahan summation algorithm, also known as compensated summation.
  Based on this [code in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/kahan.m).

  The algorithm significantly reduces the numerical error in the total
  obtained by adding a sequence of finite precision floating point numbers
  compared to the straightforward approach.  For more details
  see [this Wikipedia entry](http://en.wikipedia.org/wiki/Kahan_summation_algorithm).
  This function is called by AdaptiveStepsize.integrate to better catch
  equality comparisons.

  The first input argument is the variable that will contain the summation.
  This variable is also returned as the first output argument in order to
  reuse it in subsequent calls to `Integrator.AdaptiveStepsize.kahan_sum/3` function.

  The second input argument contains the compensation term and is returned
  as the second output argument so that it can be reused in future calls of
  the same summation.

  The third input argument `term` is the variable to be added to `sum`.
  """
  defn kahan_sum(sum, comp, term) do
    # Octave code:
    # y = term - comp;
    # t = sum + y;
    # comp = (t - sum) - y;
    # sum = t;

    y = term - comp
    t = sum + y
    comp = t - sum - y
    sum = t

    {sum, comp}
  end

  def ode_event_handler() do
  end
end
