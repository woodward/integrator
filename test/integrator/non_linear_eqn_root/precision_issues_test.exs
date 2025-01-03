defmodule Integrator.NonLinearEqnRoot.PrecisionIssuesTest do
  # Show some precision issues/problems

  @moduledoc false
  use Integrator.TestCase, async: true

  #   Works:
  #     two = Nx.tensor(2, type: type)
  #     delta = two * Nx.tensor(0.7, type: type) * (two * Nx.abs(z.u) * machine_eps + tolerance)

  # Causes problems:
  #     two = Nx.tensor(2, type: type)
  #     delta = two * 0.7 * (two * Nx.abs(z.u) * machine_eps + tolerance)

  defmodule Precision do
    @moduledoc false
    import Nx.Defn

    defn works_delta(c, u, machine_eps, tolerance) do
      type = Nx.type(c)
      two = Nx.tensor(2, type: type)
      two * Nx.tensor(0.7, type: type) * (two * Nx.abs(u) * machine_eps + tolerance)
    end

    defn does_not_work_delta(c, u, machine_eps, tolerance) do
      type = Nx.type(c)
      two = Nx.tensor(2, type: type)
      two * 0.7 * (two * Nx.abs(u) * machine_eps + tolerance)
    end

    defn two_times_point_seven_does_not_work(c) do
      type = Nx.type(c)
      two = Nx.tensor(2, type: type)
      two * 0.7
    end

    defn two_times_point_seven_works(c) do
      type = Nx.type(c)
      two = Nx.tensor(2, type: type)
      two * Nx.tensor(0.7, type: type)
    end

    defn two_times_point_five_should_not_work(c) do
      type = Nx.type(c)
      two = Nx.tensor(2, type: type)
      two * 0.5
    end

    defn two_times_point_five_works(c) do
      type = Nx.type(c)
      two = Nx.tensor(2, type: type)
      two * Nx.tensor(0.5, type: type)
    end

    defn arg_times_point_seven_works(x) do
      point_seven = Nx.tensor(0.7, type: Nx.type(x))
      x * point_seven
    end

    defn arg_times_point_seven_does_not_work(x) do
      x * 0.7
    end
  end

  describe "precision issues in InternalComputations.adjust_if_too_close_to_a_or_b/3" do
    setup do
      c = Nx.tensor(3.108624468950438e-16, type: :f64)
      u = Nx.tensor(-1.0e-8, type: :f64)
      machine_eps = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)
      [c: c, u: u, machine_eps: machine_eps, tolerance: tolerance]
    end

    test "works", %{c: c, u: u, machine_eps: machine_eps, tolerance: tolerance} do
      delta = Precision.works_delta(c, u, machine_eps, tolerance)
      assert_all_close(delta, Nx.tensor(3.1086245311229277e-16, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
    end

    @tag :skip
    test "does not work", %{c: c, u: u, machine_eps: machine_eps, tolerance: tolerance} do
      delta = Precision.does_not_work_delta(c, u, machine_eps, tolerance)
      assert_all_close(delta, Nx.tensor(3.1086245311229277e-16, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
      #                                 3.108624 4781833677e-16
    end
  end

  describe "precision issues in InternalComputations.adjust_if_too_close_to_a_or_b/3 - simple input values" do
    setup do
      c = Nx.tensor(0.1e-16, type: :f64)
      u = Nx.tensor(-1.0e-8, type: :f64)
      machine_eps = Nx.Constants.epsilon(:f64)
      tolerance = Nx.Constants.epsilon(:f64)
      [c: c, u: u, machine_eps: machine_eps, tolerance: tolerance]
    end

    test "works", %{c: c, u: u, machine_eps: machine_eps, tolerance: tolerance} do
      delta = Precision.works_delta(c, u, machine_eps, tolerance)
      assert_all_close(delta, Nx.tensor(3.1086245311229277e-16, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
    end

    @tag :skip
    test "does not work", %{c: c, u: u, machine_eps: machine_eps, tolerance: tolerance} do
      delta = Precision.does_not_work_delta(c, u, machine_eps, tolerance)
      assert_all_close(delta, Nx.tensor(3.1086245311229277e-16, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
      #                                 3.108624 4781833677e-16
      #                                 3.108624 4781833677e-16
    end
  end

  describe "2 * 0.7" do
    test "does not work if a constant" do
      c = Nx.tensor(1.0, type: :f64)
      result = Precision.two_times_point_seven_does_not_work(c)
      assert_all_close(result, Nx.tensor(1.399999976158142, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
      # assert_all_close(result, Nx.tensor(1.4, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
    end

    test "works" do
      c = Nx.tensor(1.0, type: :f64)
      result = Precision.two_times_point_seven_works(c)
      assert_all_close(result, Nx.tensor(1.4, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
    end
  end

  describe "2 * 0.5" do
    test "works even if a constant" do
      c = Nx.tensor(1.0, type: :f64)
      result = Precision.two_times_point_five_should_not_work(c)
      assert_all_close(result, Nx.tensor(1.0, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
    end

    test "works if converted to a tensor" do
      c = Nx.tensor(1.0, type: :f64)
      result = Precision.two_times_point_five_works(c)
      assert_all_close(result, Nx.tensor(1.0, type: :f64), atol: 1.0e-24, rtol: 1.0e-24)
    end
  end

  describe "using arg" do
    test "works" do
      two = Nx.tensor(2, type: :f64)
      _result = Precision.arg_times_point_seven_works(two)
      # dbg(result)
    end

    test "does not work" do
      two = Nx.tensor(2, type: :f64)
      _result = Precision.arg_times_point_seven_does_not_work(two)
      # dbg(result)
    end
  end
end
