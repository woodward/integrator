defmodule Momentum.TestCase do
  @moduledoc false

  use ExUnit.CaseTemplate

  using do
    quote do
      import Momentum.TestHelpers
    end
  end
end
