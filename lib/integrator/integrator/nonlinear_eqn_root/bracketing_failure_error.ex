defmodule Integrator.NonlinearEqnRoot.BracketingFailureError do
  @moduledoc false

  defexception message: "zero point is not bracketed",
               step: nil
end
