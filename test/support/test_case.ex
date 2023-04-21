defmodule Integrator.TestCase do
  @moduledoc false

  use ExUnit.CaseTemplate

  using do
    quote do
      import Integrator.Helpers
      import Integrator.Demo, only: [van_der_pol_fn: 2]
    end
  end
end
