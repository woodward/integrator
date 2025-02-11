defmodule Integrator.DataCollector do
  @moduledoc """
  A behaviour which defines a process which collects data from an integration
  """

  alias Integrator.Point

  @callback add_data(pid(), Point.t() | [Point.t()]) :: :ok

  @callback get_data(pid()) :: [Point.t()]

  # @callback pop_data(pid()) :: [Point.t()]

  @callback get_last_n_data(pid(), pos_integer()) :: [Point.t()]
end
