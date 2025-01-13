defmodule Integrator.NonNx.NonLinearEqnRoot.MaxIterationsExceededError do
  @moduledoc false

  defexception message: "Too many iterations",
               step: nil,
               iteration_count: nil
end
