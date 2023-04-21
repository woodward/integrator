defmodule Integrator.OdeEventHandlerTest do
  @moduledoc false
  use Integrator.DemoCase
  import Nx, only: :sigils

  alias Integrator.OdeEventHandler
  alias Integrator.AdaptiveStepsize
  alias Integrator.RungeKutta.DormandPrince45

  describe "ode_event_handler" do
    setup do
      event_fn = fn _t, x ->
        value = Nx.to_number(x[0])
        answer = if value <= 0.0, do: :halt, else: :continue
        %{status: answer, value: value}
      end

      [event_fn: event_fn]
    end

    test ":continue event", %{event_fn: event_fn} do
      t_new = 0.553549806109594
      x_new = ~V[ 1.808299104387025  -0.563813853847242 ]f64
      opts = []
      step = %AdaptiveStepsize{t_new: t_new, x_new: x_new}
      interpolate_fn_does_not_matter = & &1

      result = OdeEventHandler.call_event_fn(event_fn, step, interpolate_fn_does_not_matter, opts)

      assert result == :continue
    end

    test ":halt event for Demo.van_der_pol function (y[0] goes negative)", %{event_fn: event_fn} do
      t_old = 2.155396117711071
      t_new = 2.742956500140625
      x_old = ~V[  1.283429405203074e-02  -2.160506093425276 ]f64
      x_new = ~V[ -1.452959132853812      -2.187778875125423 ]f64

      k_vals = ~M[
          -2.160506093425276  -2.415858015466959  -2.525217131637079  -2.530906930089893  -2.373278736970216  -2.143782883869835  -2.187778875125423
          -2.172984510849814  -2.034431603317282  -1.715883769683796   2.345467244704591   3.812328420909734   4.768800180323954   3.883778892097804
        ]f64

      opts = []
      step = %AdaptiveStepsize{t_new: t_new, x_new: x_new, t_old: t_old, x_old: x_old, k_vals: k_vals}
      interpolate_fn = &DormandPrince45.interpolate/4

      {:halt, new_intermediate_step} = OdeEventHandler.call_event_fn(event_fn, step, interpolate_fn, opts)

      assert_in_delta(new_intermediate_step.t_new, 2.161317515510217, 1.0e-06)
      assert_in_delta(Nx.to_number(new_intermediate_step.x_new[0]), 2.473525941362742e-15, 1.0e-07)
      assert_in_delta(Nx.to_number(new_intermediate_step.x_new[1]), -2.173424479824061, 1.0e-07)

      assert_in_delta(Nx.to_number(new_intermediate_step.k_vals[0][0]), -2.160506093425276, 1.0e-12)
      assert new_intermediate_step.options_comp != nil
    end
  end

  describe "interpolate_one_point" do
    test "works" do
      t_old = 2.155396117711071
      t_new = 2.742956500140625
      x_old = ~V[  1.283429405203074e-02  -2.160506093425276 ]f64
      x_new = ~V[ -1.452959132853812      -2.187778875125423 ]f64

      k_vals = ~M[
            -2.160506093425276  -2.415858015466959  -2.525217131637079  -2.530906930089893  -2.373278736970216  -2.143782883869835  -2.187778875125423
            -2.172984510849814  -2.034431603317282  -1.715883769683796   2.345467244704591   3.812328420909734   4.768800180323954   3.883778892097804
          ]f64

      interpolate_fn = &DormandPrince45.interpolate/4
      step = %AdaptiveStepsize{t_new: t_new, x_new: x_new, t_old: t_old, x_old: x_old, k_vals: k_vals}

      t = 2.161317515510217
      x_interpolated = OdeEventHandler.interpolate_one_point(t, step, interpolate_fn)

      # From Octave:
      expected_x_interpolated = ~V[ 2.473525941362742e-15 -2.173424479824061  ]f64

      # Why is this not closer to tighter tolerances?
      assert_all_close(x_interpolated, expected_x_interpolated, atol: 1.0e-07, rtol: 1.0e-07)
    end
  end
end
