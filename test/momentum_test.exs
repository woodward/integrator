defmodule MomentumTest do
  @moduledoc false
  use ExUnit.Case
  doctest Momentum

  test "greets the world" do
    assert Momentum.hello() == :world
  end
end
