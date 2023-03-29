defmodule Momentum.TestCase do
  @moduledoc false

  use ExUnit.CaseTemplate

  using do
    quote do
      import Momentum.Helpers
    end
  end
end
