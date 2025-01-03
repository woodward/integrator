defmodule Integrator.TensorTypeError do
  @moduledoc false

  defexception message: "Cannot change tensor to a different type"
end
