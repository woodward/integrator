defmodule Integrator.TestCase do
  @moduledoc false

  use ExUnit.CaseTemplate, async: true

  using do
    quote do
      import Integrator.Helpers
      import Integrator.SampleEqns, only: [van_der_pol_fn: 2, euler_equations: 2]
    end
  end
end
