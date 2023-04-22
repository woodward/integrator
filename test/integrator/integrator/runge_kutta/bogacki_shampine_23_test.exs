defmodule Integrator.RungeKutta.BogackiShampine23Test do
  @moduledoc false
  use Integrator.TestCase

  # import Nx, only: :sigils
  alias Integrator.RungeKutta.BogackiShampine23

  test "order/0" do
    assert BogackiShampine23.order() == 3
  end
end
