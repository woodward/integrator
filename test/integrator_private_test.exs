defmodule IntegratorPrivateTest do
  # Test of private functions using Patch in Integrator

  @moduledoc false
  use Integrator.TestCase, async: false
  use Patch

  describe "parse_start_end/1" do
    setup do
      expose(Integrator, parse_start_end: 1)
    end

    @tag transferred_to_refactor?: false
    test "returns an array of times" do
      {t_start, t_end, fixed_times} = private(Integrator.parse_start_end([0.0, 1.0]))
      assert t_start == 0.0
      assert t_end == 1.0
      assert fixed_times == nil
    end

    @tag transferred_to_refactor?: false
    test "creates an array of fixed_times if a single tensor is given" do
      t_start = Nx.tensor(0.0, type: :f64)
      t_end = Nx.tensor(0.5, type: :f64)
      t_values = Nx.linspace(t_start, t_end, n: 6, type: :f64)

      {t_start, t_end, fixed_times} = private(Integrator.parse_start_end(t_values))
      assert t_start.__struct__ == Nx.Tensor
      assert t_start == Nx.tensor(0.0, type: :f64)
      assert Nx.type(t_start) == {:f, 64}

      assert t_end.__struct__ == Nx.Tensor
      assert t_end == Nx.tensor(0.5, type: :f64)
      assert Nx.type(t_end) == {:f, 64}

      assert length(fixed_times) == 6
      [_first_time | [second_time | _rest]] = fixed_times
      assert second_time.__struct__ == Nx.Tensor
      assert second_time == Nx.tensor(0.1, type: :f64)
      assert Nx.type(second_time) == {:f, 64}
    end
  end
end
