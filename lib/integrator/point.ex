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

  @spec points_from_t_and_x(Nx.t(), Nx.t()) :: t()
  defn points_from_t_and_x(t, x) do
    size = Nx.size(t)
    x_t = Nx.transpose(x)

    # Make this more Nx-ey somehow, and less brute-force!  There should be a general way to do this
    # that's not hard-wired to sizes 1-4. And at the very least, get a case statement to work, rather
    # than these nasty nested if statements!

    if size == 1 do
      {%__MODULE__{t: t[0], x: x_t[0]}}
    else
      if size == 2 do
        {
          %__MODULE__{t: t[0], x: x_t[0]},
          %__MODULE__{t: t[1], x: x_t[1]}
        }
      else
        if size == 3 do
          {
            %__MODULE__{t: t[0], x: x_t[0]},
            %__MODULE__{t: t[1], x: x_t[1]},
            %__MODULE__{t: t[2], x: x_t[2]}
          }
        else
          if size == 4 do
            {
              %__MODULE__{t: t[0], x: x_t[0]},
              %__MODULE__{t: t[1], x: x_t[1]},
              %__MODULE__{t: t[2], x: x_t[2]},
              %__MODULE__{t: t[3], x: x_t[3]}
            }
          else
            # Currently this is hard-wired to only go up to 4:
            raise_size_exception(size)
          end
        end
      end
    end
  end

  deftransform raise_size_exception(size) do
    raise "Tensor length #{size} is too large; only tensor lengths up to 4 are supported"
  end
end
