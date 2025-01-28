defmodule Integrator.Integration do
  @moduledoc """
  A genserver which holds the simulation state for one particular integration
  """

  use GenServer

  @impl GenServer
  def init(init_arg) do
    {:ok, init_arg}
  end
end
