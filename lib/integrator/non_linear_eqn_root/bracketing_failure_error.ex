defmodule Integrator.NonLinearEqnRoot.BracketingFailureError do
  @moduledoc false

  defexception message: "Zero point is not bracketed",
               step: nil
end
