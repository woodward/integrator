defmodule Integrator.TestCase do
  @moduledoc false

  use ExUnit.CaseTemplate

  using do
    quote do
      import Integrator.TestHelpers
    end
  end
end
