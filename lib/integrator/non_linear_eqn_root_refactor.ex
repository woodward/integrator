defmodule Integrator.NonLinearEqnRootRefactor do
  @moduledoc """
  Finds the roots (i.e., zeros) of a non-linear equation.
  Based on [`fzero.m`](https://github.com/gnu-octave/octave/blob/default/scripts/optimization/fzero.m)
  from Octave.

  This is essentially the ACM algorithm 748: Enclosing Zeros of Continuous Functions
  due to Alefeld, Potra and Shi, ACM Transactions on Mathematical Software, Vol. 21,
  No. 3, September 1995. Although the workflow is the same, the structure of
  the algorithm has been transformed non-trivially; also, the algorithm has also been
  slightly modified.
  """

  import Nx.Defn

  @derive {Nx.Container,
   containers: [
     :a,
     :b,
     :c,
     :d,
     :e,
     :u,
     #
     :fa,
     :fb,
     :fc,
     :fd,
     :fe,
     :fu,
     #
     :x,
     :fx,
     #
     :mu_ba,
     #
     :fn_eval_count,
     :iteration_count,
     :iter_type
   ],
   keep: []}

  @type t :: %__MODULE__{
          a: Nx.t(),
          b: Nx.t(),
          c: Nx.t(),
          d: Nx.t(),
          e: Nx.t(),
          u: Nx.t(),
          #
          # Function evaluations; e.g., fb is fn(b):
          fa: Nx.t(),
          fb: Nx.t(),
          fc: Nx.t(),
          fd: Nx.t(),
          fe: Nx.t(),
          fu: Nx.t(),
          #
          # x (and fx) are the actual found values (i.e., fx should be very close to zero):
          x: Nx.t(),
          fx: Nx.t(),
          #
          mu_ba: Nx.t(),
          #
          fn_eval_count: Nx.t(),
          iteration_count: Nx.t(),
          # Change iter_type to a more descriptive atom later (possibly?):
          iter_type: Nx.t()
        }

  defstruct a: 0,
            b: 0,
            c: 0,
            d: 0,
            e: 0,
            u: 0,
            #
            # Function evaluations; e.g., fb is fn(b):
            fa: 0,
            fb: 0,
            fc: 0,
            fd: 0,
            fe: 0,
            fu: 0,
            #
            # x (and fx) are the actual found values (i.e., fx should be very close to zero):
            x: 0,
            fx: 0,
            #
            mu_ba: 0,
            #
            fn_eval_count: 0,
            iteration_count: 0,
            # Change iter_type to a more descriptive atom later (possibly?):
            iter_type: 1

  @spec converged?(t(), Nx.t(), Nx.t()) :: Nx.t()
  defn converged?(z, machine_eps, tolerance) do
    if z.b - z.a <= 2 * (2 * Nx.abs(z.u) * machine_eps + tolerance) do
      halt()
    else
      continue()
    end
  end

  defnp halt(), do: 1
  defnp continue(), do: 0
end
