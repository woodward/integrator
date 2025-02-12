defmodule Integrator.DataSink do
  @moduledoc """
  Collects data from an integration (or when finding a root).
  Data is stored in reverse order in the genserver, and then its order is reversed when returned
  to the caller.
  """

  use GenServer
  use Integrator.DataCollector

  @spec start_link(Keyword.t()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, :ok, opts)
  end

  # ------------------------------------------------------------------------------------------------

  @impl true
  def init(_args) do
    {:ok, %{data: %{}}}
  end
end
