defmodule Integrator.AdaptiveStepsize.ArgPrecisionError do
  @moduledoc false

  defexception message: "argument precision error",
               invalid_argument: nil,
               argument_name: nil,
               expected_precision: nil,
               actual_precision: nil
end
