defmodule Integrator.RungeKutta.StepTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  import Nx, only: :sigils

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

      step = %Step{
        t_new: Nx.tensor(0.170323017264490, type: :f64),
        x_new: Nx.tensor([1.975376830028490, -0.266528851971234], type: :f64),
        options_comp: Nx.tensor(-1.387778780781446e-17, type: :f64),
        dt: Nx.tensor(0.153290715538041, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &DormandPrince45.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.f64(1.0e-06),
        rel_tol: Nx.f64(1.0e-03)
      }

      computed_step = Step.compute_step(step, stepper_fn, ode_fn, nx_options)

      expected_t_next = Nx.tensor(0.323613732802532, type: :f64)
      expected_x_next = Nx.tensor([1.922216228514310, -0.416811343851152], type: :f64)

      expected_k_vals = ~MAT[
        -0.266528851971234  -0.303376255443000  -0.318166975994861  -0.394383609924488  -0.412602091137911  -0.426290366186482  -0.416811343851152
        -1.201879818436319  -1.096546739499175  -1.055438526511377  -0.852388604155395  -0.804214989044028  -0.771328619755717  -0.798944990281621
      ]f64

      expected_options_comp = Nx.tensor(0.0, type: :f64)
      expected_error = Nx.tensor(1.586715304267830e-02, type: :f64)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-07, rtol: 1.0e-07)
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-07, rtol: 1.0e-07)
    end

    # Expected values were obtained from Octave for van der pol equation at t = 0.000345375551682:
    test "works - bug fix for Bogacki-Shampine23" do
      k_vals = ~MAT[
        -4.788391990136420e-04  -5.846330800545818e-04  -6.375048176907232e-04  -6.903933604135114e-04
        -1.998563425163596e+00  -1.998246018256682e+00  -1.998087382041041e+00  -1.997928701004975e+00
      ]f64

      step = %Step{
        t_new: Nx.tensor(3.453755516815583e-04, type: :f64),
        x_new: Nx.tensor([1.999999880756917, -6.903933604135114e-04], type: :f64),
        options_comp: Nx.tensor(1.355252715606881e-20, type: :f64),
        dt: Nx.tensor(1.048148240128353e-04, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      }

      computed_step = Step.compute_step(step, stepper_fn, ode_fn, nx_options)

      expected_t_next = Nx.tensor(4.501903756943936e-04, type: :f64)
      expected_x_next = Nx.tensor([1.999999797419839, -8.997729805855904e-04], type: :f64)

      expected_k_vals = ~MAT[
        -6.903933604135114e-04  -7.950996330065260e-04  -8.474280732402655e-04  -8.997729805855904e-04
        -1.997928701004975e+00  -1.997614546170481e+00  -1.997457534649594e+00  -1.997300479207187e+00
      ]f64

      expected_options_comp = Nx.tensor(-2.710505431213761e-20, type: :f64)
      expected_error = Nx.tensor(0.383840528805912, type: :f64)

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-16, rtol: 1.0e-16)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-17, rtol: 1.0e-17)

      # Note that the error is just accurate to single precision, which is ok; see the test below for abs_rel_norm
      # to see how sensitive the error is to input values:
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-07, rtol: 1.0e-07)
    end

    # Expected values and inputs were obtained from Octave for van der pol equation at t = 0.000239505625605:
    test "works - bug fix for Bogacki-Shampine23 - 2nd attempt" do
      k_vals = ~MAT[
        -2.585758248155079e-04  -3.687257174741470e-04  -4.237733527882678e-04  -4.788391990136420e-04
        -1.999224255823159e+00  -1.998893791926987e+00  -1.998728632828801e+00  -1.998563425163596e+00
      ]f64

      step = %Step{
        t_new: Nx.tensor(2.395056256047516e-04, type: :f64),
        x_new: Nx.tensor([1.999999942650792, -4.788391990136420e-04], type: :f64),
        options_comp: Nx.tensor(-1.355252715606881e-20, type: :f64),
        dt: Nx.tensor(1.058699260768067e-04, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      }

      computed_step = Step.compute_step(step, stepper_fn, ode_fn, nx_options)

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

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-15, rtol: 1.0e-15)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-17, rtol: 1.0e-17)
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-15, rtol: 1.0e-15)
    end

    # Inputs were obtained from AdaptiveStepsize for van der pol equation at t = 0.000239505625605:
    test "works - bug fix for Bogacki-Shampine23 - 2nd attempt - using inputs from Elixir, not Octave" do
      # Expected values are from Octave
      k_vals = ~MAT[
        -2.585758248155079e-4  -3.687257174741469e-4  -4.2377335278826774e-4  -4.78839199013642e-4
        -1.999224255823159     -1.9988937919269867    -1.9987286328288005     -1.9985634251635955
      ]f64

      step = %Step{
        t_new: Nx.tensor(2.3950562560475164e-04, type: :f64),
        x_new: Nx.tensor([1.9999999426507922, -4.78839199013642e-4], type: :f64),
        options_comp: Nx.tensor(0.0, type: :f64),
        dt: Nx.tensor(1.0586992285952218e-4, type: :f64),
        # dt is WRONG!!!!
        # dt: Nx.tensor(1.058699260768067e-04, type: :f64),
        k_vals: k_vals
      }

      stepper_fn = &BogackiShampine23.integrate/6
      ode_fn = &SampleEqns.van_der_pol_fn/2

      nx_options = %NxOptions{
        norm_control?: Nx.u8(0),
        abs_tol: Nx.tensor(1.0e-12, type: :f64),
        rel_tol: Nx.tensor(1.0e-12, type: :f64)
      }

      computed_step = Step.compute_step(step, stepper_fn, ode_fn, nx_options)

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

      assert_all_close(computed_step.t_new, expected_t_next, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.x_new, expected_x_next, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.k_vals, expected_k_vals, atol: 1.0e-11, rtol: 1.0e-11)
      assert_all_close(computed_step.options_comp, expected_options_comp, atol: 1.0e-19, rtol: 1.0e-19)
      assert_all_close(computed_step.error_estimate, expected_error, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end
end
