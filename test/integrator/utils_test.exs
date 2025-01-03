defmodule Integrator.UtilsTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  import Nx.Defn
  import Nx, only: :sigils

  alias Integrator.TensorTypeError
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

  describe "same_signs?/2" do
    setup do
      one = Nx.u8(1)
      zero = Nx.u8(0)
      [zero: zero, one: one]
    end

    test "returns 1 if both quantities have the same sign", %{one: one} do
      assert Utils.same_signs?(Nx.f32(-2.0), Nx.f32(-4.0)) == one
      assert Utils.same_signs?(Nx.f32(3.0), Nx.f32(7.0)) == one
    end

    test "returns 0 if both quantities have different signs", %{zero: zero} do
      assert Utils.same_signs?(Nx.f32(-2.0), Nx.f32(4.0)) == zero
      assert Utils.same_signs?(Nx.f32(3.0), Nx.f32(-7.0)) == zero
    end

    test "returns 0 if one quantity is zero", %{zero: zero} do
      assert Utils.same_signs?(Nx.f32(-2.0), Nx.f32(0.0)) == zero
      assert Utils.same_signs?(Nx.f32(0.0), Nx.f32(0.0)) == zero
    end

    test "returns 0 if both quantities are zero", %{zero: zero} do
      assert Utils.same_signs?(Nx.f32(0.0), Nx.f32(0.0)) == zero
    end
  end

  describe "same_signs_or_any_zeros?/2" do
    setup do
      one = Nx.u8(1)
      zero = Nx.u8(0)
      [zero: zero, one: one]
    end

    test "returns 1 if both quantities have the same sign", %{one: one} do
      assert Utils.same_signs_or_any_zeros?(Nx.f32(-2.0), Nx.f32(-4.0)) == one
      assert Utils.same_signs_or_any_zeros?(Nx.f32(3.0), Nx.f32(7.0)) == one
    end

    test "returns 0 if both quantities have different signs", %{zero: zero} do
      assert Utils.same_signs_or_any_zeros?(Nx.f32(-2.0), Nx.f32(4.0)) == zero
      assert Utils.same_signs_or_any_zeros?(Nx.f32(3.0), Nx.f32(-7.0)) == zero
    end

    test "returns 1 if one quantity is zero", %{one: one} do
      assert Utils.same_signs_or_any_zeros?(Nx.f32(-2.0), Nx.f32(0.0)) == one
      assert Utils.same_signs_or_any_zeros?(Nx.f32(0.0), Nx.f32(0.0)) == one
    end

    test "returns 1 if both quantities are zero", %{one: one} do
      assert Utils.same_signs_or_any_zeros?(Nx.f32(0.0), Nx.f32(0.0)) == one
    end
  end

  describe "different_signs_or_any_zeros?/2" do
    setup do
      one = Nx.u8(1)
      zero = Nx.u8(0)
      [zero: zero, one: one]
    end

    test "returns 0 if both quantities have the same sign", %{zero: zero} do
      assert Utils.different_signs_or_any_zeros?(Nx.f32(-2.0), Nx.f32(-4.0)) == zero
      assert Utils.different_signs_or_any_zeros?(Nx.f32(3.0), Nx.f32(7.0)) == zero
    end

    test "returns 1 if both quantities have different signs", %{one: one} do
      assert Utils.different_signs_or_any_zeros?(Nx.f32(-2.0), Nx.f32(4.0)) == one
      assert Utils.different_signs_or_any_zeros?(Nx.f32(3.0), Nx.f32(-7.0)) == one
    end

    test "returns 1 if one quantity is zero", %{one: one} do
      assert Utils.different_signs_or_any_zeros?(Nx.f32(-2.0), Nx.f32(0.0)) == one
      assert Utils.different_signs_or_any_zeros?(Nx.f32(0.0), Nx.f32(0.0)) == one
    end

    test "returns 1 if both quantities are zero", %{one: one} do
      assert Utils.different_signs_or_any_zeros?(Nx.f32(0.0), Nx.f32(0.0)) == one
    end
  end

  describe "different_signs?/2" do
    setup do
      one = Nx.u8(1)
      zero = Nx.u8(0)
      [zero: zero, one: one]
    end

    test "returns 0 if both quantities have the same sign", %{zero: zero} do
      assert Utils.different_signs?(Nx.f32(-2.0), Nx.f32(-4.0)) == zero
      assert Utils.different_signs?(Nx.f32(3.0), Nx.f32(7.0)) == zero
    end

    test "returns 1 if the two quantities have different signs", %{one: one} do
      assert Utils.different_signs?(Nx.f32(-2.0), Nx.f32(4.0)) == one
      assert Utils.different_signs?(Nx.f32(3.0), Nx.f32(-7.0)) == one
    end

    test "returns 0 if one quantity is zero", %{zero: zero} do
      assert Utils.different_signs?(Nx.f32(-2.0), Nx.f32(0.0)) == zero
      assert Utils.different_signs?(Nx.f32(0.0), Nx.f32(0.0)) == zero
    end

    test "returns 0 if both quantities are zero", %{zero: zero} do
      assert Utils.different_signs?(Nx.f32(0.0), Nx.f32(0.0)) == zero
    end
  end

  describe "convert_arg_to_nx_type" do
    test "passes through tensors (if they are of the correct type)" do
      arg = Nx.f64(1.0)
      assert Utils.convert_arg_to_nx_type(arg, {:f, 64}) == Nx.f64(1.0)
    end

    test "converts floats to tensors of the appropriate type" do
      arg = 1.0
      assert Utils.convert_arg_to_nx_type(arg, {:f, 32}) == Nx.f32(1.0)

      assert Utils.convert_arg_to_nx_type(arg, {:f, 64}) == Nx.f64(1.0)
    end

    test "allows functions to pass through" do
      arg = &Nx.sin/1
      assert Utils.convert_arg_to_nx_type(arg, {:f, 64}) == (&Nx.sin/1)
    end

    test "raises an exception if you try to cast a tensor to a different type" do
      arg = Nx.f64(1.0)

      assert_raise TensorTypeError, fn ->
        Utils.convert_arg_to_nx_type(arg, {:f, 32})
      end
    end
  end

  describe "abs_rel_norm/6" do
    # These test values were obtained from Octave:
    test "when norm_control: Nx.tensor(false)" do
      t = Nx.tensor([1.97537683003, -0.26652885197])
      t_old = Nx.tensor([1.99566026409, -0.12317664679])
      abs_tolerance = 1.0000e-06
      rel_tolerance = 1.0000e-03
      norm_control = Nx.tensor(false)
      x = Nx.tensor([1.97537723429, -0.26653011403])
      expected_norm = Nx.tensor(0.00473516383083)

      norm = Utils.abs_rel_norm(t, t_old, x, abs_tolerance, rel_tolerance, norm_control)

      assert_all_close(norm, expected_norm, atol: 1.0e-04, rtol: 1.0e-04)
    end

    test "when norm_control: Nx.tensor(false) - :f64 - starting_stepsize for high-fidelity ballode" do
      x0 = ~VEC[  0.0 20.0  ]f64
      abs_tol = Nx.tensor(1.0e-14, type: :f64)
      rel_tol = Nx.tensor(1.0e-14, type: :f64)
      norm_control = Nx.tensor(false)
      x_zeros = Nx.tensor([0.0, 0.0], type: :f64)

      norm = Utils.abs_rel_norm(x0, x0, x_zeros, abs_tol, rel_tol, norm_control)

      assert_all_close(norm, Nx.tensor(1.0e14, type: :f64), atol: 1.0e-17, rtol: 1.0e-17)
    end

    # All values taken from Octave for the high-fidelity Bogacki-Shampine23 at t = 0.000345375551682:
    test "when norm_control: Nx.tensor(false) - :f64 - for high-fidelity Bogacki-Shampine" do
      x_old = ~VEC[ 1.999999880756917  -6.903933604135114e-04 ]f64
      #         [ 1.999999880756917, -6.903933604135114e-04 ]  Elixir values agree exactly

      x_next = ~VEC[ 1.999999797419839   -8.997729805855904e-04  ]f64
      #          [ 1.9999997974198394, -8.997729805855904e-4]  Elixir values agree exactly

      # This works (from Octave):
      x_est = ~VEC[ 1.999999797419983  -8.997729809694310e-04 ]f64

      # This doesn't work (from Elixir); note the _very_ small differences in x[1]:
      # x_est = ~VEC[ 1.9999997974199832 -8.997729809694309e-04 ]f64
      # x_est = ~VEC[ 1.999999797419983  -8.997729809694310e-04 ]f64  Octave values from above

      # From Octave:
      expected_error = Nx.tensor(0.383840528805912, type: :f64)
      #                          0.3838404203856949
      #                          Value from Elixir using x_est above.
      # Note that it seems to be just single precision agreement
      # The equations in abs_rel_norm check out ok; they are just SUPER sensitive to small differences
      # in the input values

      abs_tol = Nx.tensor(1.0e-12, type: :f64)
      rel_tol = Nx.tensor(1.0e-12, type: :f64)

      error = Utils.abs_rel_norm(x_next, x_old, x_est, abs_tol, rel_tol, Nx.tensor(false))

      assert_all_close(error, expected_error, atol: 1.0e-16, rtol: 1.0e-16)
    end

    # Octave:
    test "when norm_control: Nx.tensor(false) - :f64 - for test 'works - high fidelity - playback speed of 0.5'" do
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-11, "RelTol", 1.0e-11, "Refine", 1);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      # Values for t = 0.005054072392284442:
      x_next = ~VEC[ 1.9998466100062002  -0.024463877616688966 ]f64
      # Elixir:
      # x_next = ~VEC[ 1.999846610006200 2  -0.0244638776166889 66 ]f64
      # Octave:
      # x_next = ~VEC[ 1.999846610006200    -0.0244638776166889 7 ]f64

      x_old = ~VEC[  1.9999745850165596  -0.010031858255616163 ]f64
      # Octave:
      # x_old = ~VEC[  1.999974585016560   -0.01003185825561616 ]f64

      x_est = ~VEC[  1.9998466100068684  -0.024463877619281038 ]f64
      # Elixir:
      # x_est = ~VEC[  1.999846610006868 4  -0.0244638776192810 38 ]f64
      # Octave:
      # x_est = ~VEC[  1.999846610006868    -0.0244638776192810 4 ]f64

      # Octave:

      abs_tol = Nx.tensor(1.0e-11, type: :f64)
      rel_tol = Nx.tensor(1.0e-11, type: :f64)

      expected_error = Nx.tensor(0.259206892061492, type: :f64)

      # error = Utils.abs_rel_norm(x_next, x_old, x_est, abs_tol, rel_tol, norm_control: Nx.tensor(false))
      {error, _t_minus_x} = abs_rel_norm_for_test_purposes(x_next, x_old, x_est, abs_tol, rel_tol, norm_control: Nx.tensor(false))

      # IO.inspect(Nx.to_number(error), label: "error")
      # IO.inspect(t_minus_x, label: "t_minus_x")

      # sc (Elixir): [1.9999745850165594e-11, 1.0e-11
      # sc (Octave): [1.999974585016559e-11   9.999999999999999e-12]

      # x_next - x_est, which is t - x in abs_rel_norm:
      # t - x (Elixir): [-6.681322162194192e-13, 2.5920 723900618725e-12]
      # t - x (Octave): [-6.681322162194192e-13, 2.5920 68920614921e-12]   SINGLE PRECISION AGREEMENT!!!

      # Nx.abs(t - x) (Elixir) [6.681322162194192e-13, 2.5920 723900618725e-12]
      # Nx.abs(t - x) (Octave) [6.681322162194192e-13, 2.5920 68920614921e-12 ]  SINGLE PRECISION AGREEMENT!!!

      # We can currently get single precision agreement, but not double precision:
      assert_all_close(error, expected_error, atol: 1.0e-06, rtol: 1.0e-06)

      _subtraction = Nx.subtract(x_next[1], x_est[1])
      # IO.inspect(Nx.to_number(subtraction), label: "problematic subtraction")
      # subtraction (Elixir): 2.5920 723900618725e-12
      # subtraction (Octave): 2.5920 68920614921e-12

      # This doesn't work, but should:
      # assert_all_close(error, expected_error, atol: 1.0e-11, rtol: 1.0e-11)
    end

    defn abs_rel_norm_for_test_purposes(t, t_old, x, abs_tolerance, rel_tolerance, _opts \\ []) do
      # Octave code:
      #   sc = max (AbsTol(:), RelTol .* max (abs (x), abs (x_old)));
      #   retval = max (abs (x - y) ./ sc);

      sc = Nx.max(abs_tolerance, rel_tolerance * Nx.max(Nx.abs(t), Nx.abs(t_old)))
      {(Nx.abs(t - x) / sc) |> Nx.reduce_max(), t - x}
    end

    # Values from Octave:
    test "trying to figure out precision problem" do
      x_new_2 = Nx.tensor(-2.446387761668897e-02, type: :f64)
      x_est_2 = Nx.tensor(-2.446387761928104e-02, type: :f64)

      # problematic subtraction         : 2.5920 723900618725e-12
      # expected_subtraction_from_octave: 2.5920 68920614921e-12

      # If I enter these values directly into Octave (rather than printing them out from the integration proces)
      # I get:
      # Octave:
      #   x_new_2 = -2.446387761668897e-02
      #   x_est_2 = -2.446387761928104e-02
      #   subtraction = x_new_2 - x_est_2
      #   2.592072390061873e-12  Octave value
      #   2.5920723900618725e-12 Elixir value
      # which is also single precision agreement

      subtraction = Nx.subtract(x_new_2, x_est_2)
      # IO.inspect(Nx.to_number(subtraction), label: "problematic subtraction         ")
      expected_subtraction_from_octave = Nx.tensor(2.592068920614921e-12, type: :f64)
      #                                            2.5920 72390061873e-12  Octave from above when directly entering values
      # IO.inspect(Nx.to_number(expected_subtraction_from_octave), label: "expected_subtraction_from_octave")
      # assert_all_close(subtraction, expected_subtraction_from_octave, atol: 1.0e-06, rtol: 1.0e-06)
      assert_all_close(subtraction, expected_subtraction_from_octave, atol: 1.0e-17, rtol: 1.0e-17)
    end

    # All values taken from Octave from test "works - high fidelity - playback speed of 0.5" for the 2nd timestep
    test "when norm_control: Nx.tensor(false) - :f64 - for high-fidelity van der pol" do
      # Octave:
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-11, "RelTol", 1.0e-11);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      x_old = ~VEC[ 1.999974585016560   -0.01003185825561616  ]f64
      # xoe = ~VEC[ 1.9999745850165596  -0.010031858255616163 ]f64  Elixir values agree exactly

      x_next = ~VEC[ 1.999846610006200   -0.02446387761668897 ]f64
      # xne = ~VEC[  1.9998466100062002  -0.024463877616688966 ]f64  Elixir values agree exactly

      # This works (from Octave):
      x_est = ~VEC[ 1.999846610006868   -0.02446387761928104  ]f64

      # This doesn't work (from Elixir); note the _very_ small differences in x[1]:
      # x_est = ~VEC[ 1.9998466100068684  -0.024463877619281038 ]f64
      # x_est = ~VEC[ 1.999846610006868   -0.02446387761928104  ]f64  Octave values from above to compare

      # From Octave:
      expected_error = Nx.tensor(0.259206892061492, type: :f64)
      #                          0.2592072390061872
      #                          Value from Elixir using x_est above.
      # Note that it seems to be just single precision agreement
      # The equations in abs_rel_norm check out ok; they are just SUPER sensitive to small differences
      # in the input values

      abs_tol = Nx.tensor(1.0e-11, type: :f64)
      rel_tol = Nx.tensor(1.0e-11, type: :f64)

      error = Utils.abs_rel_norm(x_next, x_old, x_est, abs_tol, rel_tol, Nx.tensor(false))

      # sc:   [1.99997458501656e-11, 1.0e-11]  Elixir                Agreement!!!
      # sc:    1.999974585016559e-11 9.999999999999999e-12 Octave

      # t:  [1.9998466100062,  -0.02446387761668897]  Elixir       Agreement!!
      # t:   1.999846610006200 -0.02446387761668897   Octave

      # x:  [1.999846610006868, -0.02446387761928104]  Elixir     Agreement!!
      # x:   1.999846610006868  -0.02446387761928104   Octave

      # t - x:   [-6.6 79101716144942e-13, 2.5920 723900618725e-12]  Elixir  Not so great :(
      # t - x:    -6.6 81322162194192e-13  2.5920 68920614921e-12    Octave

      # Nx.abs(t - x):  [6.679101716144942e-13, 2.5920723900618725e-12]  Elixir
      # Nx.abs(t - x):   6.681322162194192e-13  2.592068920614921e-12    Octave

      # Nx.abs(t - x) / sc:  [0.033 39593295926627, 0.25920 723900618725]  Elixir
      # Nx.abs(t - x) / sc:   0.033 40703533059582  0.25920 68920614921    Octave
      assert_all_close(error, expected_error, atol: 1.0e-06, rtol: 1.0e-06)

      # Should be able to get this precision:
      # assert_all_close(error, expected_error, atol: 1.0e-16, rtol: 1.0e-16)
    end

    # These test values were obtained from Octave:
    test "when norm_control: Nx.tensor(true)" do
      x = Nx.tensor([1.99465419035, 0.33300240425])
      x_old = Nx.tensor([1.64842646336, 1.78609260054])
      abs_tolerance = 1.0000e-06
      rel_tolerance = 1.0000e-03
      norm_control = Nx.tensor(true)
      y = Nx.tensor([1.99402286380, 0.33477644992])
      expected_norm = Nx.tensor(0.77474409123)

      norm = Utils.abs_rel_norm(x, x_old, y, abs_tolerance, rel_tolerance, norm_control)

      assert_all_close(norm, expected_norm, atol: 1.0e-04, rtol: 1.0e-04)
    end
  end
end
