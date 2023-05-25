defmodule Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError do
  @moduledoc false

  defexception message: "Too many function evaluations",
               step: nil,
               fn_eval_count: nil
end
