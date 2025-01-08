defmodule Integrator.Point do
  @moduledoc """
  Strips out the t and x values from a `RungeKutta.Step` into a form that can be used for various
  output purposes
  """

  import Nx.Defn

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

  @spec split_points_into_t_and_x([Point.t()]) :: {[Nx.t()], [Nx.t()]}
  def split_points_into_t_and_x(points) do
    t = points |> Enum.map(& &1.t)
    x = points |> Enum.map(& &1.x)
    {t, x}
  end

  @spec convert_to_points(Nx.t(), Nx.t()) :: t()
  defn convert_to_points(t, x) do
    x_t = Nx.transpose(x)

    # Make this more Nx-ey somehow, and less brute-force!  There should be a general way to do this
    # that's not hard-wired to sizes 1-4. But how???
    case Nx.size(t) do
      1 ->
        {%__MODULE__{t: t[0], x: x_t[0]}}

      2 ->
        {
          %__MODULE__{t: t[0], x: x_t[0]},
          %__MODULE__{t: t[1], x: x_t[1]}
        }

      3 ->
        {
          %__MODULE__{t: t[0], x: x_t[0]},
          %__MODULE__{t: t[1], x: x_t[1]},
          %__MODULE__{t: t[2], x: x_t[2]}
        }

      4 ->
        {
          %__MODULE__{t: t[0], x: x_t[0]},
          %__MODULE__{t: t[1], x: x_t[1]},
          %__MODULE__{t: t[2], x: x_t[2]},
          %__MODULE__{t: t[3], x: x_t[3]}
        }

      size ->
        # Currently this function is hard-wired to only allow up to size 4:
        raise_size_exception(size)
    end
  end

  deftransform raise_size_exception(size) do
    raise "Tensor length #{size} is too large; only tensor lengths up to 4 are supported"
  end
end
