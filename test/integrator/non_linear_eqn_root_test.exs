defmodule Integrator.NonLinearEqnRootTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  import Nx, only: :sigils

  alias Integrator.DataCollector
  alias Integrator.NonLinearEqnRoot
  alias Integrator.NonLinearEqnRoot.InternalComputations
  alias Integrator.NonLinearEqnRoot.InvalidInitialBracketError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError
  alias Integrator.NonLinearEqnRoot.NonLinearEqnRootOptions
  alias Integrator.RungeKutta.DormandPrince45

  defmodule TestFunctions do
    @moduledoc false
    import Nx.Defn

    defn pow_fn(x, _args) do
      type = Nx.type(x)
      power = Nx.tensor(1, type: type) / Nx.tensor(3, type: type)
      Nx.pow(x, power) - Nx.tensor(1.0e-8, type: type)
    end

    defn straight_line_through_zero(x, _args), do: x
    defn straight_line_offset_by_one(x, _args), do: x + 1

    defn polynomial(x, _args) do
      x * x - 4 * x + 3
    end

    defn sin(x, _args), do: Nx.sin(x)
    defn cos(x, _args), do: Nx.cos(x)

    defn ballode(t_out, args) do
      # Values obtained from Octave right before and after the call to fzero in ode_event_handler.m:
      [t0, t1, x0, x1, k_vals] = args
      t = Nx.concatenate([t0, t1])
      x = Nx.stack([x0, x1]) |> Nx.transpose()
      x_out = DormandPrince45.interpolate(t, x, k_vals, t_out)
      x_out[0][0]
    end
  end

  describe "find_zero" do
    test "sine function (so the zeros of this are known values) - computations in :f64" do
      # Octave:
      # fun = @sin; % function
      # x0 = 3;
      # x1 = 4;
      # x = fzero(fun, [x0, x1])
      # x = 3.141592653589795
      #
      # opts = optimset("Display", "iter")
      # x = fzero(fun, [x0, x1], opts)

      # Search for a zero in the interval [3, 4]:
      #  Fcn-count    x          f(x)             Procedure
      #     2              3       0.14112        initial
      #     3        3.15716    -0.0155695        interpolation
      #     4        3.14128   0.000310917        interpolation
      #     5        3.14159    3.9018e-08        interpolation
      #     6        3.14159  -3.90211e-08        interpolation
      #     7        3.14159   1.22465e-16        interpolation
      #     8        3.14159  -2.09798e-15        interpolation
      # Algorithm converged.
      # x = 3.141592653589795

      x0 = Nx.f64(3.0)
      x1 = Nx.f64(4.0)

      result = NonLinearEqnRoot.find_zero(&TestFunctions.sin/2, x0, x1, [])

      # Expected value is from Octave:
      expected_x = Nx.f64(3.141592653589795)
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, 0.0, atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.fn_eval_count) == 8
      assert Nx.to_number(result.iteration_count) == 6
      assert Nx.to_number(result.iteration_type) == 4

      {x_low, x_high} = InternalComputations.bracket_x(result)
      # Expected values are from Octave:
      assert_all_close(x_low, Nx.f64(3.141592653589793), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.f64(3.141592653589795), atol: 1.0e-14, rtol: 1.0e-14)

      {y1, y2} = InternalComputations.bracket_fx(result)
      # Expected values are from Octave:
      assert_all_close(y1, Nx.f64(1.224646799147353e-16), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.f64(-2.097981369335578e-15), atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.elapsed_time_μs) > 0
    end

    test "sine function - works if initial values are swapped" do
      x0 = Nx.f64(4.0)
      x1 = Nx.f64(3.0)

      result = NonLinearEqnRoot.find_zero(&TestFunctions.sin/2, x0, x1, [])

      # Expected value is from Octave:
      expected_x = Nx.f64(3.141592653589795)
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, Nx.f64(0.0), atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.fn_eval_count) == 8
      assert Nx.to_number(result.iteration_count) == 6
      assert Nx.to_number(result.iteration_type) == 4

      {x_low, x_high} = InternalComputations.bracket_x(result)
      # Expected values are from Octave:
      assert_all_close(x_low, Nx.f64(3.141592653589793), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.f64(3.141592653589795), atol: 1.0e-14, rtol: 1.0e-14)

      {y1, y2} = InternalComputations.bracket_fx(result)
      # Expected values are from Octave:
      assert_all_close(y1, Nx.f64(1.224646799147353e-16), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.f64(-2.097981369335578e-15), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "sine function - raises an error if invalid initial bracket - positive sine" do
      # Sine is positive for both of these:
      x0 = 2.5
      x1 = 3.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRoot.find_zero(&TestFunctions.sin/2, x0, x1, [])
      end
    end

    test "sine function - raises an error if invalid initial bracket - negative sine" do
      # Sine is negative for both of these:
      x0 = 3.5
      x1 = 4.0

      assert_raise InvalidInitialBracketError, fn ->
        NonLinearEqnRoot.find_zero(&TestFunctions.sin/2, x0, x1, [])
      end
    end

    test "sine function - raises an error if max iterations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_iterations: 2]

      assert_raise MaxIterationsExceededError, fn ->
        NonLinearEqnRoot.find_zero(&TestFunctions.sin/2, x0, x1, [], opts)
      end
    end

    test "sine function - raises an error if max function evaluations exceeded" do
      x0 = 3.0
      x1 = 4.0
      opts = [max_fn_eval_count: 2]

      assert_raise MaxFnEvalsExceededError, fn ->
        NonLinearEqnRoot.find_zero(&TestFunctions.sin/2, x0, x1, [], opts)
      end
    end

    test "sine function - outputs values if a function is given" do
      # Octave:
      #   octave> fun = @sin;
      #   octave> x0 = 3;
      #   octave> x1 = 4;
      #   octave> x = fzero(fun, [x0, x1])

      x0 = Nx.f64(3.0)
      x1 = Nx.f64(4.0)

      {:ok, pid} = DataCollector.start_link()
      output_fn = &DataCollector.add_data(pid, &1)

      opts = [nonlinear_eqn_root_output_fn: output_fn]

      result = NonLinearEqnRoot.find_zero(&TestFunctions.sin/2, x0, x1, [], opts)
      assert_all_close(result.x, Nx.f64(3.1415926535897936), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, Nx.f64(-3.216245299353273e-16), atol: 1.0e-14, rtol: 1.0e-14)

      data = DataCollector.get_data(pid)
      assert length(data) == 6

      # From Octave:
      converging_t_data = [
        Nx.f64(3.157162792479947),
        Nx.f64(3.141281736699444),
        Nx.f64(3.141592614571824),
        Nx.f64(3.141592692610915),
        Nx.f64(3.141592653589793),
        Nx.f64(3.141592653589795)
      ]

      t_data = data |> Enum.map(& &1.x)

      assert_nx_lists_equal_refactor(t_data, converging_t_data)
      expected_t = converging_t_data |> Enum.reverse() |> hd()
      assert_all_close(result.x, expected_t, atol: 1.0e-14, rtol: 1.0e-14)

      converged = data |> List.last()

      assert Nx.to_number(converged.iteration_count) == 6
      assert Nx.to_number(converged.fn_eval_count) == 8
      assert_all_close(converged.x, result.x, atol: 1.0e-14, rtol: 1.0e-14)
    end
  end

  describe "convert_to_nx_compatible_options" do
    test "uses the defaults from nimble options (and defaults for machine_eps and tolerance in the type specified)" do
      opts = []

      nx_options = NonLinearEqnRoot.convert_to_nx_compatible_options(opts)
      assert %NonLinearEqnRootOptions{} = nx_options

      assert_all_close(nx_options.machine_eps, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.machine_eps) == {:f, 64}

      assert_all_close(nx_options.tolerance, Nx.Constants.epsilon(:f64), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.tolerance) == {:f, 64}

      assert nx_options.type == {:f, 64}
      assert nx_options.max_iterations == 1_000
      assert nx_options.max_fn_eval_count == 1_000
      assert nx_options.nonlinear_eqn_root_output_fn == nil
    end

    test "allows overrides" do
      output_fn = fn x -> x end

      opts = [
        type: :f32,
        max_iterations: 10,
        max_fn_eval_count: 20,
        machine_eps: 0.1,
        tolerance: 0.2,
        nonlinear_eqn_root_output_fn: output_fn
      ]

      nx_options = NonLinearEqnRoot.convert_to_nx_compatible_options(opts)
      assert %NonLinearEqnRootOptions{} = nx_options

      assert_all_close(nx_options.machine_eps, Nx.f32(0.1), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.machine_eps) == {:f, 32}

      assert_all_close(nx_options.tolerance, Nx.f32(0.2), atol: 1.0e-16, rtol: 1.0e-16)
      assert Nx.type(nx_options.tolerance) == {:f, 32}

      assert nx_options.type == {:f, 32}
      assert nx_options.max_iterations == 10
      assert nx_options.max_fn_eval_count == 20
      assert nx_options.nonlinear_eqn_root_output_fn == output_fn
    end

    test "sine function with single initial value (instead of 2)" do
      x0 = Nx.f64(3.0)

      result = NonLinearEqnRoot.find_zero_with_single_point(&TestFunctions.sin/2, x0, [])

      # Expected value is from Octave:
      expected_x = Nx.f64(3.141592653589795)
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(result.fx, Nx.f64(0.0), atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.fn_eval_count) == 11
      assert Nx.to_number(result.iteration_count) == 4
      assert Nx.to_number(result.iteration_type) == 2

      {x_low, x_high} = InternalComputations.bracket_x(result)
      # Expected values are from Octave:
      assert_all_close(x_low, Nx.f64(3.141592653589793), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.f64(3.141592653589795), atol: 1.0e-14, rtol: 1.0e-14)

      {y1, y2} = InternalComputations.bracket_fx(result)
      # Expected values are from Octave:
      assert_all_close(y1, Nx.f64(1.224646799147353e-16), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.f64(-2.097981369335578e-15), atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "returns pi/2 for cos between 0 & 3 - test from Octave" do
      x0 = Nx.f64(0.0)
      x1 = Nx.f64(3.0)

      result = NonLinearEqnRoot.find_zero(&TestFunctions.cos/2, x0, x1, [])

      expected_x = Nx.divide(Nx.Constants.pi({:f, 64}), Nx.f64(2.0))
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
    end

    test "equation - test from Octave" do
      # Octave (this code is at the bottom of fzero.m):
      #   fun = @(x) x^(1/3) - 1e-8
      #   fzero(fun, [0.0, 1.0])
      x0 = Nx.f64(0.0)
      x1 = Nx.f64(1.0)
      zero_fn = &TestFunctions.pow_fn/2

      result = NonLinearEqnRoot.find_zero(zero_fn, x0, x1, [])

      # Expected values are from Octave:
      assert_all_close(result.x, Nx.f64(3.108624468950438e-16), atol: 1.0e-24, rtol: 1.0e-24)
      assert_all_close(result.fx, Nx.f64(6.764169935169993e-06), atol: 1.0e-22, rtol: 1.0e-22)
    end

    test "staight line through zero - test from Octave" do
      # Octave (this code is at the bottom of fzero.m):
      #   fun = @(x) x
      #   fzero(fun, 0)
      x0 = Nx.f64(0.0)
      zero_fn = &TestFunctions.straight_line_through_zero/2

      result = NonLinearEqnRoot.find_zero_with_single_point(zero_fn, x0, [])

      assert_all_close(result.x, Nx.f64(0.0), atol: 1.0e-24, rtol: 1.0e-24)
      assert_all_close(result.fx, Nx.f64(0.0), atol: 1.0e-24, rtol: 1.0e-24)
    end

    test "staight line through zero offset by one - test from Octave" do
      x0 = Nx.f64(0.0)
      zero_fn = &TestFunctions.straight_line_offset_by_one/2

      result = NonLinearEqnRoot.find_zero_with_single_point(zero_fn, x0, [])

      assert_all_close(result.x, Nx.f64(-1.0), atol: 1.0e24, rtol: 1.0e24)
      assert_all_close(result.fx, Nx.f64(0.0), atol: 1.0e24, rtol: 1.0e24)
    end

    test "polynomial" do
      # y = (x - 1) * (x - 3) = x^2 - 4*x + 3
      # Roots are 1 and 3

      zero_fn = &TestFunctions.polynomial/2

      result = NonLinearEqnRoot.find_zero(zero_fn, 0.5, 1.5, [], type: :f64)

      assert_all_close(result.x, Nx.f64(1.0), atol: 1.0e24, rtol: 1.0e24)
      assert_all_close(result.fx, Nx.f64(0.0), atol: 1.0e24, rtol: 1.0e24)

      result = NonLinearEqnRoot.find_zero(zero_fn, 3.5, 1.5, [], type: :f64)

      assert_all_close(result.x, Nx.f64(3.0), atol: 1.0e24, rtol: 1.0e24)
      assert_all_close(result.fx, Nx.f64(0.0), atol: 1.0e24, rtol: 1.0e24)
    end

    test "ballode - first bounce" do
      t0 = Nx.f64([2.898648469921000])
      t1 = Nx.f64([4.294180317944318])

      x0 = Nx.f64([1.676036011799988e+01, -8.435741489925014e+00])
      x1 = Nx.f64([-4.564518118928532e+00, -2.212590891903376e+01])

      k_vals = ~MAT[
          -8.435741489925014e+00  -1.117377497574676e+01  -1.254279171865764e+01  -1.938787543321202e+01  -2.060477920468836e+01   -2.212590891903378e+01  -2.212590891903376e+01
          -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00  -9.810000000000000e+00   -9.810000000000000e+00  -9.810000000000000e+00
      ]f64

      zero_fn = &TestFunctions.ballode/2
      zero_fn_args = [t0, t1, x0, x1, k_vals]

      # Same values as in ballode definition above:
      t0 = Nx.f64(2.898648469921000)
      t1 = Nx.f64(4.294180317944318)

      result = NonLinearEqnRoot.find_zero(zero_fn, t0, t1, zero_fn_args)

      # Expected value is from Octave:
      expected_x = Nx.f64(4.077471967380223)
      assert_all_close(result.c, expected_x, atol: 1.0e-14, rtol: 1.0e-14)
      # This should be close to zero because we found the zero root:
      assert_all_close(result.fx, Nx.f64(0.0), atol: 1.0e-14, rtol: 1.0e-14)

      assert Nx.to_number(result.fn_eval_count) == 7
      assert Nx.to_number(result.iteration_count) == 5
      assert Nx.to_number(result.iteration_type) == 3

      {x__low, x_high} = InternalComputations.bracket_x(result)
      # Expected values are from Octave; note that these are the same except in the last digit:
      assert_all_close(x__low, Nx.f64(4.077471967380224), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(x_high, Nx.f64(4.077471967380227), atol: 1.0e-14, rtol: 1.0e-14)
      # Octave:
      # 4.077471967380223
      # 4.077471967380223

      {y_1, y2} = InternalComputations.bracket_fx(result)
      assert_all_close(y_1, Nx.f64(0.0), atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(y2, Nx.f64(0.0), atol: 1.0e-14, rtol: 1.0e-14)
      # In Octave:
      # [0, 0]

      x_out = TestFunctions.ballode(result.c, zero_fn_args)
      assert_all_close(Nx.to_number(x_out), Nx.f64(0.0), atol: 1.0e-14, rtol: 1.0e-14)
    end
  end

  describe "bracket_x/1" do
    test "returns a & b" do
      z = %NonLinearEqnRoot{
        a: 3.14,
        b: 3.15
      }

      assert InternalComputations.bracket_x(z) == {3.14, 3.15}
    end
  end

  describe "bracket_fx/1" do
    test "returns fa & fb" do
      z = %NonLinearEqnRoot{
        fa: 3.14,
        fb: 3.15
      }

      assert InternalComputations.bracket_fx(z) == {3.14, 3.15}
    end
  end

  describe "option_keys" do
    test "returns the option keys" do
      assert NonLinearEqnRoot.option_keys() == [
               :nonlinear_eqn_root_output_fn,
               :type,
               :max_fn_eval_count,
               :max_iterations
             ]
    end
  end
end
