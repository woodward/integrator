defmodule Integrator.NonLinearEqnRoot.InvalidInitialBracketError do
  @moduledoc false

  defexception message: "Invalid initial bracket",
               step: nil
end
