defmodule Integrator.RungeKutta.StepTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  import Nx, only: :sigils

  alias Integrator.AdaptiveStepsizeRefactor
  alias Integrator.AdaptiveStepsizeRefactor.NxOptions
  alias Integrator.RungeKutta.BogackiShampine23
  alias Integrator.RungeKutta.DormandPrince45
  alias Integrator.RungeKutta.Step
  alias Integrator.SampleEqns

  describe "compute_step" do
    # Expected values were obtained from Octave:
    test "works" do
      k_vals = ~MAT[
        -0.123176646786029  -0.156456392653781  -0.170792108688503  -0.242396950166743  -0.256398564600740  -0.270123280961810  -0.266528851971234
        -1.628266220377807  -1.528057633442594  -1.484796318238127  -1.272143242010950  -1.231218923718637  -1.191362260138565  -1.201879818436319
      ]f64

      dt = Nx.tensor(0.153290715538041, type: :f64)

      step = %Step{
        t_new: Nx.tensor(0.170323017264490, type: :f64),
        x_new: Nx.tensor([1.975376830028490, -0.266528851971234], type: :f64),
        options_comp: Nx.tensor(-1.387778780781446e-17, type: :f64),
        #
        # Setting dt to something to verify that its value is overridden (plus it needs to be non-nil to cross the defn boundary)
        dt: Nx.f64(0.0),
        #
        k_vals: k_vals,
        #
        # Not used, but these need to be non-zero in order to cross the Elixir-Nx boundary:
        x_old: Nx.f64(0.0),
        t_old: Nx.f64(0.0),
        error_estimate: Nx.f64(0.0)
      }

      stepper_fn = &DormandPrince45.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03)
      }

      computed_step = Step.compute_step(step, dt, stepper_fn, ode_fn, nx_options)

      expected_t_next = Nx.tensor(0.323613732802532, type: :f64)
      expected_x_next = Nx.tensor([1.922216228514310, -0.416811343851152], type: :f64)

      expected_k_vals = ~MAT[
        -0.266528851971234  -0.303376255443000  -0.318166975994861  -0.394383609924488  -0.412602091137911  -0.426290366186482  -0.416811343851152
        -1.201879818436319  -1.096546739499175  -1.055438526511377  -0.852388604155395  -0.804214989044028  -0.771328619755717  -0.798944990281621
      ]f64

      expected_options_comp = Nx.tensor(0.0, type: :f64)
      expected_error = Nx.tensor(1.586715304267830e-02, type: :f64)

      assert_all_close(computed_step.t_old, step.t_new, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(computed_step.x_old, step.x_new, atol: 1.0e-14, rtol: 1.0e-14)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-07, rtol: 1.0e-07)

      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-07, rtol: 1.0e-07)

      assert computed_step.dt == dt
    end

    # Expected values were obtained from Octave for van der pol equation at t = 0.000345375551682:
    test "works - bug fix for Bogacki-Shampine23" do
      k_vals = ~MAT[
        -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04
        -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00
      ]f64

      dt = Nx.tensor(1.048148240128353e-04, type: :f64)

      step = %Step{
        t_new: Nx.tensor(3.453755516815583e-04, type: :f64),
        x_new: Nx.tensor([1.999999880756917, -6.903933604135114e-04], type: :f64),
        options_comp: Nx.tensor(1.355252715606881e-20, type: :f64),
        #
        # Setting dt to something to verify that its value is overridden (plus it needs to be non-nil to cross the defn boundary)
        dt: Nx.f64(0.0),
        #
        k_vals: k_vals,
        #
        # Not used, but these need to be non-zero in order to cross the Elixir-Nx boundary:
        x_old: Nx.f64(0.0),
        t_old: Nx.f64(0.0),
        error_estimate: Nx.f64(0.0)
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      }

      computed_step = Step.compute_step(step, dt, stepper_fn, ode_fn, nx_options)

      expected_t_next = Nx.tensor(4.501903756943936e-04, type: :f64)
      expected_x_next = Nx.tensor([1.999999797419839, -8.997729805855904e-04], type: :f64)

      expected_k_vals = ~MAT[
        -6.903933604135114e-04  -7.950996330065260e-04  -8.474280732402655e-04  -8.997729805855904e-04
        -1.997928701004975e+00  -1.997614546170481e+00  -1.997457534649594e+00  -1.997300479207187e+00
      ]f64

      expected_options_comp = Nx.tensor(-2.710505431213761e-20, type: :f64)
      expected_error = Nx.tensor(0.383840528805912, type: :f64)

      assert_all_close(computed_step.t_old, step.t_new, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(computed_step.x_old, step.x_new, atol: 1.0e-14, rtol: 1.0e-14)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)

      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-17, rtol: 1.0e-17)

      # Note that the error is just accurate to single precision, which is ok; see the test below for abs_rel_norm
      # to see how sensitive the error is to input values:
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-07, rtol: 1.0e-07)

      assert computed_step.dt == dt
    end

    # Expected values and inputs were obtained from Octave for van der pol equation at t = 0.000239505625605:
    test "works - bug fix for Bogacki-Shampine23 - 2nd attempt" do
      k_vals = ~MAT[
        -2.585758248155079e-04  -3.687257174741470e-04  -4.237733527882678e-04  -4.788391990136420e-04
        -1.999224255823159e+00  -1.998893791926987e+00  -1.998728632828801e+00  -1.998563425163596e+00
      ]f64

      dt = Nx.tensor(1.058699260768067e-04, type: :f64)

      step = %Step{
        t_new: Nx.tensor(2.395056256047516e-04, type: :f64),
        x_new: Nx.tensor([1.999999942650792, -4.788391990136420e-04], type: :f64),
        options_comp: Nx.tensor(-1.355252715606881e-20, type: :f64),
        #
        # Setting dt to something to verify that its value is overridden (plus it needs to be non-nil to cross the defn boundary)
        dt: Nx.f64(0.0),
        #
        k_vals: k_vals,
        #
        # Not used, but these need to be non-zero in order to cross the Elixir-Nx boundary:
        x_old: Nx.f64(0.0),
        t_old: Nx.f64(0.0),
        error_estimate: Nx.f64(0.0)
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      }

      computed_step = Step.compute_step(step, dt, stepper_fn, ode_fn, nx_options)

      expected_t_next = Nx.tensor(3.453755516815583e-04, type: :f64)
      #                   Elixir: 3.4537555168155827e-4
      expected_x_next = Nx.tensor([1.999999880756917, -6.903933604135114e-04], type: :f64)
      #                  Elixir:  [1.9999998807569168 -6.903933604135114e-4]

      expected_k_vals = ~MAT[
        -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04
        -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00
      ]f64

      # -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04  Octave
      # -4.78839199013642e-4    -5.846330800545818e-4   -6.375048176907232e-4   -6.903933604135114e-4],  Elixir

      # -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00  Octave
      # -1.998563425163596,     -1.9982460182566815,    -1.998087382041041,     -1.9979287010049749]    Elixir
      expected_options_comp = Nx.tensor(1.355252715606881e-20, type: :f64)
      #                       Elixir:  -2.710505431213761e-20
      expected_error = Nx.tensor(0.395533432395734, type: :f64)
      #                  Elixir: 0.3955334323957338

      # dbg(error)

      assert_all_close(computed_step.t_old, step.t_new, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(computed_step.x_old, step.x_new, atol: 1.0e-14, rtol: 1.0e-14)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)

      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-15, rtol: 1.0e-15)

      assert computed_step.dt == dt
    end

    # Inputs were obtained from AdaptiveStepsize for van der pol equation at t = 0.000239505625605:
    test "works - bug fix for Bogacki-Shampine23 - 2nd attempt - using inputs from Elixir, not Octave" do
      # Expected values are from Octave
      k_vals = ~MAT[
        -2.585758248155079e-4  -3.687257174741469e-4  -4.2377335278826774e-4  -4.78839199013642e-4
        -1.999224255823159     -1.9988937919269867    -1.9987286328288005     -1.9985634251635955
      ]f64

      # dt is WRONG!!!!
      # dt: Nx.tensor(1.058699260768067e-04, type: :f64),
      dt = Nx.tensor(1.0586992285952218e-4, type: :f64)

      step = %Step{
        t_new: Nx.tensor(2.3950562560475164e-04, type: :f64),
        x_new: Nx.tensor([1.9999999426507922, -4.78839199013642e-4], type: :f64),
        options_comp: Nx.tensor(0.0, type: :f64),
        #
        # Setting dt to something to verify that its value is overridden (plus it needs to be non-nil to cross the defn boundary)
        dt: Nx.f64(0.0),
        #
        k_vals: k_vals,
        #
        # Not used, but these need to be non-zero in order to cross the Elixir-Nx boundary:
        x_old: Nx.f64(0.0),
        t_old: Nx.f64(0.0),
        error_estimate: Nx.f64(0.0)
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      }

      computed_step = Step.compute_step(step, dt, stepper_fn, ode_fn, nx_options)

      expected_t_next = Nx.tensor(3.453755516815583e-04, type: :f64)
      #                   Elixir: 3.453755 484642738e-4
      expected_x_next = Nx.tensor([1.999999880756917, -6.903933604135114e-04], type: :f64)
      #                  Elixir:  [1.9999998807569193 -6.903933 539856063e-4]

      expected_k_vals = ~MAT[
        -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04
        -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00
      ]f64

      # -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04  Octave
      # -4.78839199013642e-4    -5.846330800545818e-4   -6.375048176907232e-4   -6.903933604135114e-4],  Elixir

      # -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00  Octave
      # -1.998563425163596,     -1.9982460182566815,    -1.998087382041041,     -1.9979287010049749]    Elixir
      expected_options_comp = Nx.tensor(1.355252715606881e-20, type: :f64)
      #                       Elixir:  -2.710505431213761e-20
      expected_error = Nx.tensor(0.395533432395734, type: :f64)
      #                  Elixir: 0.3955334323957338

      assert_all_close(computed_step.t_old, step.t_new, atol: 1.0e-14, rtol: 1.0e-14)
      assert_all_close(computed_step.x_old, step.x_new, atol: 1.0e-14, rtol: 1.0e-14)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-11, rtol: 1.0e-11)

      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-19, rtol: 1.0e-19)
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-15, rtol: 1.0e-15)

      assert computed_step.dt == dt
    end
  end

  describe "initial_step/?" do
    test "returns an inital step with the correct initial values" do
      t0 = Nx.f64(1.0)
      x0 = Nx.f64([2.0, 3.0])
      order = 5

      initial_step = Step.initial_step(t0, x0, order: order)

      expected_k_vals = ~MAT[
        0.0  0.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0
      ]f64

      nan = Nx.Constants.nan(:f64)

      assert initial_step == %Step{
               t_old: nan,
               t_new: Nx.f64(1.0),
               #
               x_old: Nx.tensor([0.0, 0.0], type: :f64),
               x_new: Nx.f64([2.0, 3.0]),
               #
               k_vals: expected_k_vals,
               options_comp: Nx.f64(0.0),
               error_estimate: nan,
               dt: nan
             }
    end
  end

  describe "initial_output_t_and_x" do
    test "returns t and x of the correct sizes and type - no interpolation nor fixed times" do
      options = %AdaptiveStepsizeRefactor.NxOptions{refine: 1, fixed_output_times?: Nx.u8(0), type: {:f, 64}}
      x0 = Nx.f32([0.0, 0.0])

      {output_t, output_x} = Step.initial_output_t_and_x(x0, options)

      assert output_t == Nx.f64([0.0])
      assert output_x == Nx.f64([0.0, 0.0])
    end

    test "returns t and x of the correct sizes and type - fixed times (which means no interpolation)" do
      options = %AdaptiveStepsizeRefactor.NxOptions{refine: 1, fixed_output_times?: Nx.u8(1), type: {:f, 64}}
      x0 = Nx.f32([0.0, 0.0])

      {output_t, output_x} = Step.initial_output_t_and_x(x0, options)

      assert output_t == Nx.f64([0.0])
      assert output_x == Nx.f64([0.0, 0.0])
    end

    test "returns t and x of the correct sizes and type - interpolation and no fixed times" do
      options = %AdaptiveStepsizeRefactor.NxOptions{refine: 4, fixed_output_times?: Nx.u8(0), type: {:f, 64}}
      x0 = Nx.f32([0.0, 0.0])

      {output_t, output_x} = Step.initial_output_t_and_x(x0, options)

      assert output_t == Nx.f64([0.0, 0.0, 0.0, 0.0])
      assert output_x == ~MAT[
        0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0
      ]f64
    end
  end

  # %__MODULE__{
  #   t_new: t_start,
  #   x_new: x0,
  #   # t_old must be set on the initial struct in case there's an error when computing the first step (used in t_next/2)
  #   t_old: t_start,
  #   dt: initial_tstep,
  #   k_vals: initial_empty_k_vals(order, x0),
  #   fixed_times: fixed_times,
  #   nx_type: nx_type,
  #   options_comp: Nx.tensor(0.0, type: nx_type),
  #   timestamp_μs: timestamp_now,
  #   timestamp_start_μs: timestamp_now
  # }

  describe "initial_empty_k_vals - defn version" do
    test "returns a tensor with zeros that's the correct size, given the size of x and the order" do
      order = 5
      x = ~VEC[ 1.0 2.0 3.0 ]f64
      k_vals = Step.initial_empty_k_vals_defn(x, order: order)

      expected_k_vals = ~MAT[
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
      ]f64

      assert_all_close(k_vals, expected_k_vals, atol: 1.0e-15, rtol: 1.0e-16)

      # The k_vals has the Nx type of x:
      assert Nx.type(k_vals) == {:f, 64}
    end

    test "returns a tensor that has the Nx type of x" do
      order = 3
      type = {:f, 32}
      x = Nx.tensor([1.0, 2.0, 3.0], type: type)
      k_vals = Step.initial_empty_k_vals_defn(x, order: order)

      expected_k_vals = ~MAT[
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
      ]f32

      assert_all_close(k_vals, expected_k_vals, atol: 1.0e-15, rtol: 1.0e-16)

      # The k_vals has the Nx type of x:
      assert Nx.type(k_vals) == type
    end
  end

  describe "initial_empty_k_vals - deftransform version" do
    test "returns a tensor with zeros that's the correct size, given the size of x and the order" do
      order = 5
      x = ~VEC[ 1.0 2.0 3.0 ]f64
      k_vals = Step.initial_empty_k_vals(order, x)

      expected_k_vals = ~MAT[
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0
      ]f64

      assert_all_close(k_vals, expected_k_vals, atol: 1.0e-15, rtol: 1.0e-16)

      # The k_vals has the Nx type of x:
      assert Nx.type(k_vals) == {:f, 64}
    end

    test "returns a tensor that has the Nx type of x" do
      order = 3
      type = {:f, 32}
      x = Nx.tensor([1.0, 2.0, 3.0], type: type)
      k_vals = Step.initial_empty_k_vals(order, x)

      expected_k_vals = ~MAT[
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
      ]f32

      assert_all_close(k_vals, expected_k_vals, atol: 1.0e-15, rtol: 1.0e-16)

      # The k_vals has the Nx type of x:
      assert Nx.type(k_vals) == type
    end
  end

  describe "interpolate_multiple_points" do
    test "returns the multiple interpolated points (without the first point, but with the last)" do
      # These test values were obtained from Octave:
      # Generated using:
      # fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      # opts = odeset("AbsTol", 1.0e-14, "RelTol", 1.0e-14)
      # [t,x] = ode45 (fvdp, [0, 20], [2, 0], opts);

      t_old = Nx.f64(19.97226029930081)
      t_new = Nx.f64(19.97424839002798)
      x_old = Nx.f64([2.008585111348593e+00, 1.188547490189183e-02])
      x_new = Nx.f64([2.008604708104012e+00, 7.832739209072674e-03])

      der = ~MAT[
        1.188547490189183e-02   1.107248473635211e-02   1.066709096215445e-02   8.641324907205110e-03   8.281808253394873e-03     7.832711009917654e-03   7.832739209072674e-03
       -2.044650564564792e+00  -2.042188551791212e+00  -2.040960496435665e+00  -2.034823361858414e+00  -2.033733968850626e+00    -2.032373024665618e+00  -2.032373099413282e+00
      ]f64

      expected_x_out = ~MAT[
       2.008590766279272e+00   2.008595916876415e+00   2.008600563898753e+00   2.008604708104012e+00
       1.087000165112446e-02   9.856055965852775e-03   8.843635825513105e-03   7.832739209072674e-03
      ]f64

      expected_t_out = ~VEC[ 19.97275732198261   19.97325434466440   19.97375136734619   19.97424839002798 ]f64

      rk_step = %Step{
        t_old: t_old,
        t_new: t_new,
        x_old: x_old,
        x_new: x_new,
        k_vals: der,
        #
        # These values just need to be something other than nil:
        options_comp: Nx.f64(0.0),
        error_estimate: Nx.f64(0.0),
        dt: Nx.f64(0.0)
      }

      interpolate_fn = &DormandPrince45.interpolate/4
      options = %NxOptions{refine: 4, type: {:f, 64}}

      {t_out, x_out} = Step.interpolate_multiple_points(interpolate_fn, rk_step, options)

      assert_all_close(t_out, expected_t_out, atol: 1.0e-14, rtol: 1.0e-14)

      assert_all_close(x_out, expected_x_out, atol: 1.0e-13, rtol: 1.0e-13)
      assert_nx_f64(x_out)
    end
  end
end
