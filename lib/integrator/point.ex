defmodule Integrator.Point do
  @moduledoc """
  Strips out the t and x values from a `RungeKutta.Step` into a form that can be used for various
  output purposes
  """

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

  @spec split_points_into_t_and_x([t()]) :: {[Nx.t()], [Nx.t()]}
  def split_points_into_t_and_x(points) do
    t = points |> Enum.map(& &1.t)
    x = points |> Enum.map(& &1.x)
    {t, x}
  end

  def to_number(point) do
    t = Nx.to_number(point.t)
    x = Nx.to_list(point.x) |> Enum.map(&Nx.to_number(&1))
    %__MODULE__{t: t, x: x}
  end

  # This was used in the MultiIntegrator tests; a point was being broadcast twice (at the end of one
  # integration and the start of the next).  I couldn't figure out a better solution, so this is sort
  # of a temporary patch/hack.
  def filter_out_points_with_same_t(points) do
    points
    |> Enum.reduce({-1, []}, fn point, {last_t, points} ->
      if last_t != point.t do
        {point.t, [point | points]}
      else
        {last_t, points}
      end
    end)
    |> elem(1)
    |> Enum.reverse()
  end
end
