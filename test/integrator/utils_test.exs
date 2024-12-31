defmodule Integrator.UtilsTest do
  @moduledoc false
  use Integrator.TestCase

  alias Integrator.Utils

  describe "kahan_sum" do
    test "sums up some items" do
      sum = Nx.tensor(2.74295650014, type: :f64)
      comp = Nx.tensor(1.11022302463e-16, type: :f64)
      term = Nx.tensor(0.66059601818, type: :f64)

      expected_sum = Nx.tensor(3.40355251832, type: :f64)
      expected_comp = Nx.tensor(1.11022302463e-16, type: :f64)

      {sum, comp} = Utils.kahan_sum(sum, comp, term)

      assert_all_close(sum, expected_sum, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(comp, expected_comp, atol: 1.0e-14, rtol: 1.0e-14)
      assert_nx_f64(sum)
      assert_nx_f64(comp)
    end

    test "another test case" do
      # All values are taken from Octave:
      t_old = Nx.tensor(3.636484156979396e-02, type: :f64)
      options_comp_old = Nx.tensor(3.469446951953614e-18, type: :f64)
      dt = Nx.tensor(8.037014854361582e-03, type: :f64)

      {t_new, options_comp_new} = Utils.kahan_sum(t_old, options_comp_old, dt)

      expected_t_new = Nx.tensor(4.440185642415553e-02, type: :f64)
      #                          4.4401856424155534e-02  From Elixir
      # IO.inspect(Nx.to_number(t_new), label: "t_new")

      expected_options_comp_new = Nx.tensor(-1.734723475976807e-18, type: :f64)
      #                                     -1.734723475976807e-18  From Elixir
      # IO.inspect(Nx.to_number(options_comp_new), label: "options_comp_new")

      assert_all_close(t_new, expected_t_new, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(options_comp_new, expected_options_comp_new, atol: 1.0e-23, rtol: 1.0e-23)
    end
  end

  describe "columns_as_list" do
    test "works" do
      matrix = Nx.iota({2, 5})
      cols_as_list = Utils.columns_as_list(matrix, 1, 3)

      expected_cols_as_list = [
        # Nx.tensor([0, 5]),
        Nx.tensor([1, 6]),
        Nx.tensor([2, 7]),
        Nx.tensor([3, 8])
        # Nx.tensor([4, 9]),
      ]

      assert cols_as_list == expected_cols_as_list
    end

    test "goes all the way to the end if the end_index is left out" do
      matrix = Nx.iota({2, 5})
      cols_as_list = Utils.columns_as_list(matrix, 1)

      expected_cols_as_list = [
        # Not present: Nx.tensor([0, 5]),
        Nx.tensor([1, 6]),
        Nx.tensor([2, 7]),
        Nx.tensor([3, 8]),
        Nx.tensor([4, 9])
      ]

      assert cols_as_list == expected_cols_as_list
    end
  end

  describe "vector_as_list" do
    test "works" do
      vector = Nx.tensor([1, 2, 3], type: :f64)
      vector_as_list = vector |> Utils.vector_as_list()

      assert vector_as_list == [
               Nx.tensor(1, type: :f64),
               Nx.tensor(2, type: :f64),
               Nx.tensor(3, type: :f64)
             ]

      [first | [second | [third]]] = vector_as_list
      assert_nx_f64(first)
      assert_nx_f64(second)
      assert_nx_f64(third)
    end
  end

  describe "sign" do
    test "is negative one for things less than one" do
      assert Utils.sign(-7.0) == -1.0
    end

    test "is plus one for things less than one" do
      assert Utils.sign(7.0) == 1.0
    end

    test "is zero for zero" do
      assert Utils.sign(0.0) == 0.0
    end
  end

  describe "unique/1" do
    test "returns all values if they are unique" do
      assert Utils.unique([3.3, 2.2, 1.1]) == [1.1, 2.2, 3.3]
    end

    test "only returns unique values sorted in ascending order" do
      assert Utils.unique([3.3, 2.2, 3.3]) == [2.2, 3.3]
    end
  end
end
