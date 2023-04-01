defmodule IntegratorTest do
  @moduledoc false
  use ExUnit.Case
  doctest Integrator

  test "greets the world" do
    assert Integrator.hello() == :world
  end
end
