defmodule Integrator.OdeEventHandlerTest do
  @moduledoc false
  use Integrator.DemoCase
  import Nx, only: :sigils

  alias Integrator.OdeEventHandler

  describe "ode_event_handler" do
    setup do
      nx_true = Nx.tensor(1, type: :u8)

      event_fn = fn _t, x ->
        answer = Nx.less_equal(x[0], Nx.tensor(0.0)) == nx_true
        answer = if answer, do: :halt, else: :continue
        %{status: answer}
      end

      [event_fn: event_fn]
    end

    test ":continue event", %{event_fn: event_fn} do
      # ode_fn = &Demo.van_der_pol_fn/2

      t = 0.553549806109594
      x = ~V[ 1.808299104387025  -0.563813853847242 ]f64

      k_vals =
        ~M[ -0.416811343851152  -0.453552598621893  -0.467529481245710  -0.544559818448525  -0.569007867819576  -0.582739802332717  -0.563813853847242
            -0.798944990281621  -0.714018105104644  -0.686266735910648  -0.545285740580080  -0.494564963280272  -0.478302356501250  -0.528472298914133
        ]f64

      order = 5
      opts = []

      result = OdeEventHandler.call_event_fn(event_fn, t, x, k_vals, order, opts)

      assert result == %OdeEventHandler{status: :continue}
    end

    test ":halt event", %{event_fn: event_fn} do
      # ode_fn = &Demo.van_der_pol_fn/2

      t = 2.742956500140625
      x = ~V[ -1.452959132853812  -2.187778875125423 ]f64

      k_vals = ~M[
          -2.160506093425276  -2.415858015466959  -2.525217131637079  -2.530906930089893  -2.373278736970216  -2.143782883869835  -2.187778875125423
          -2.172984510849814  -2.034431603317282  -1.715883769683796   2.345467244704591   3.812328420909734   4.768800180323954   3.883778892097804
        ]f64

      order = 5
      opts = []

      result = OdeEventHandler.call_event_fn(event_fn, t, x, k_vals, order, opts)

      assert result == %OdeEventHandler{status: :halt}
    end
  end

  describe "fzero plus interpolate" do
    test "works" do
      t_old = 2.155396117711071
      t_new = 2.742956500140625
      y_old = ~V[ 1.283429405203074e-02 -2.160506093425276e+00 ]f64
      y_new = ~V[ -1.452959132853812 -2.187778875125423 ]f64

      #  call some function here

      t_zero = 2.161317515510217
      y_zero = ~V[ 2.473525941362742e-15  -2.173424479824061e+00 ]f64
    end
  end
end
