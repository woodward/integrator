defmodule Integrator.DataCollector do
  @moduledoc """
  A behaviour which defines a process which collects data from an integration
  """

  alias Integrator.Point

  @callback add_data(data_sink_pid :: pid(), data :: Point.t() | [Point.t()]) :: :ok

  @callback get_data(data_sink_pid :: pid()) :: [Point.t()]

  @callback pop_data(data_sink_pid :: pid()) :: [Point.t()]

  @callback get_last_n_data(data_sink_pid :: pid(), n_points :: pos_integer()) :: [Point.t()]
end
