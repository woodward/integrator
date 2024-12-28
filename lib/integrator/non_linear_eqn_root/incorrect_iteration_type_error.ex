defmodule Integrator.NonLinearEqnRoot.IncorrectIterationTypeError do
  @moduledoc false

  defexception message: "Invalid/incorrect iteration type",
               step: nil,
               iter_type: nil
end
