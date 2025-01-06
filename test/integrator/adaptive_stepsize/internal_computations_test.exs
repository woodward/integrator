defmodule Integrator.InternalComputations.InternalComputationsTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.AdaptiveStepsize.InternalComputations

  describe "compute_next_timestep" do
    test "basic case" do
      dt = Nx.f64(0.068129)
      error = Nx.f64(0.0015164936598390992)
      order = 5
      t_old = Nx.f64(0.0)
      t_end = Nx.f64(2.0)
      opts = [type: :f64, max_step: 2.0]

      new_dt = InternalComputations.compute_next_timestep(dt, error, order, t_old, t_end, opts)

      expected_dt = Nx.f64(0.1022)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "uses option :max_step" do
      dt = Nx.f64(0.068129)
      error = Nx.f64(0.0015164936598390992)
      order = 5
      t_old = Nx.f64(0.0)
      t_end = Nx.f64(2.0)
      opts = [max_step: 0.05, type: :f64]

      new_dt = InternalComputations.compute_next_timestep(dt, error, order, t_old, t_end, opts)

      expected_dt = Nx.f64(0.05)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "does not go past t_end" do
      dt = Nx.f64(0.3039)
      error = Nx.f64(0.4414)
      order = 5
      t_old = Nx.f64(19.711)
      t_end = Nx.f64(20.0)
      opts = [type: :f64, max_step: 2.0]

      new_dt = InternalComputations.compute_next_timestep(dt, error, order, t_old, t_end, opts)

      expected_dt = Nx.f64(0.289)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-05, rtol: 1.0e-05)
    end

    test "bug fix for Bogacki-Shampine high fidelity (see high fidelity Bogacki-Shampine test above)" do
      dt = Nx.f64(2.020515504676623e-4)
      error = Nx.f64(2.7489475539627106)
      order = 3
      t_old = Nx.f64(0.0)
      t_end = Nx.f64(20.0)
      opts = [type: :f64, max_step: Nx.f64(2.0)]

      new_dt = InternalComputations.compute_next_timestep(dt, error, order, t_old, t_end, opts)

      # From Octave:
      expected_dt = Nx.f64(1.616412403741299e-04)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-19, rtol: 1.0e-19)
    end

    # Octave:
    test "2nd bug fix for Bogacki-Shampine high fidelity (see high fidelity Bogacki-Shampine test above) - compare Elixir input" do
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-12, "RelTol", 1.0e-12, "Refine", 1);
      #   [t,x] = ode23 (fvdp, [0, 0.1], [2, 0], opts);

      # Input values are from Elixir for t_old = 2.395056256047516e-04:
      dt = Nx.f64(1.1019263330544775e-04)
      #              1.101926333054478e-04  Octave

      # If I use error value from Octave - succeeds:
      error = Nx.f64(0.445967698534111)

      # TRY ENABLING THIS AGAIN AFTER KAHAN FIX!!!
      #  If I use error value from Elixir - fails:
      # error = Nx.f64(0.4459677527442196)

      order = 3
      t_old = Nx.f64(2.3950562560475164e-04)
      t_end = Nx.f64(0.1)
      opts = [type: :f64, max_step: Nx.f64(2.0)]

      new_dt = InternalComputations.compute_next_timestep(dt, error, order, t_old, t_end, opts)

      # Expected dt from Octave:
      expected_dt = Nx.f64(1.058699260768067e-04)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-19, rtol: 1.0e-19)
    end

    # Octave:
    test "bug fix for 'works - high fidelity - playback speed of 0.5'" do
      #   format long
      #   fvdp = @(t,x) [x(2); (1 - x(1)^2) * x(2) - x(1)];
      #   opts = odeset("AbsTol", 1.0e-11, "RelTol", 1.0e-11, "Refine", 1);
      #   [t,x] = ode45 (fvdp, [0, 0.1], [2, 0], opts);

      # Input values are from Elixir for t_old = 0.005054072392284442:
      dt = Nx.f64(0.007408247469735083)
      #              0.007408247469735083  Octave  EXACT MATCH!

      # Error from Elixir only agrees to single precision - THIS IS THE PROBLEM!
      # error_from_elixir = Nx.f64(0.25920723900618725)

      # The test passes if I use the error from Octave:
      error_from_octave = Nx.f64(0.259206892061492)
      # error = error_from_elixir
      error = error_from_octave

      order = 5
      t_old = Nx.f64(0.012462319862019525)
      #                 0.01246231986201952   Octave  EXACT MATCH!

      t_end = Nx.f64(0.1)

      opts = [
        type: :f64,
        max_step: Nx.f64(0.01),
        rel_tol: Nx.f64(1.0e-11),
        abs_tol: Nx.f64(1.0e-11),
        norm_control: false
      ]

      new_dt = InternalComputations.compute_next_timestep(dt, error, order, t_old, t_end, opts)

      # Expected dt from Octave:
      expected_dt = Nx.f64(0.007895960916517373)
      assert_all_close(new_dt, expected_dt, atol: 1.0e-19, rtol: 1.0e-19)
    end
  end
end
