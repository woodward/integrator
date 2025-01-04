defmodule Integrator.Point do
  @moduledoc """
  Strips out the t and x values from a `RungeKutta.Step` into a form that can be used for various
  output purposes
  """

  import Nx.Defn

  alias Integrator.Utils

  @derive {Nx.Container,
           containers: [
             :t,
             :x
           ]}

  @type t :: %__MODULE__{
          t: Nx.t(),
          x: Nx.t()
        }

  defstruct [
    :t,
    :x
  ]

  @spec points_from_t_and_x(Nx.t(), Nx.t()) :: t()
  deftransform points_from_t_and_x(t, x) do
    # Change this to a defn somehow and use Nx properly!

    x_cols = Utils.columns_as_list(x, 0)
    t_list = Nx.to_list(t)

    Enum.zip(t_list, x_cols)
    |> Enum.map(fn {t, x} ->
      %__MODULE__{t: t, x: Nx.to_list(x)}
    end)
  end

  defn what_does_this_do?(x) do
    xt = x |> Nx.transpose()
    {xt[0], xt[1]}
  end
end
