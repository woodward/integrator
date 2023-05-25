defmodule Integrator.AdaptiveStepsize.MaxErrorsExceededError do
  @moduledoc false

  defexception message: "Too many errors",
               error_count: 0,
               max_number_of_errors: 0,
               step: nil
end
