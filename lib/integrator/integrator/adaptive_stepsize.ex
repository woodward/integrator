defmodule Integrator.AdaptiveStepsize do
  @moduledoc false
  import Nx.Defn
  alias Integrator.Utils

  defmodule TempResults do
    @moduledoc false
    defstruct [
      :dt,
      :factor,
      :k_vals
    ]
  end

  defmodule StepAccumulator do
    @moduledoc false
    defstruct [
      :t_old,
      :t_new,
      :x_old,
      :x_new,
      :dt,
      :k_vals,
      count_loop: 0,
      count_cycles: 0,
      count_save: 2,
      i_reject: 0,
      i_step: 0,
      options_comp: 0.0,
      unhandled_termination: true,
      terminal_event: false,
      terminal_output: false,
      ode_t: [],
      ode_x: [],
      output_x: [],
      output_t: []
    ]
  end

  defstruct [
    :output_t,
    :output_y,
    #
    # Temp results:
    :temp,
    #
    count_loop: 0,
    count_cycles: 0,
    count_save: 2,
    unhandled_termination: true
  ]

  # @factor_min 0.8
  # @factor_max 1.5

  # Is refine from the ode order? and should not be hard-wired?
  @refine 4

  @nx_true Nx.tensor(1, type: :u8)
  # @nx_false Nx.tensor(1, type: :u8)

  @doc """

  See [Wikipedia](https://en.wikipedia.org/wiki/Adaptive_stepsize)
  """
  def integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts \\ []) do
    dt = initial_tstep

    # Formula taken from Hairer
    factor = Math.pow(0.38, 1.0 / (order + 1))

    temp = %TempResults{dt: dt, factor: factor}

    # norm_control = false

    # i_out = 1
    # i_step = 1

    # t_new = t_start
    # t_old = t_start
    # ode_t = t_start
    # output_t = t_start

    # x_new = x0
    # x_old = x0
    # ode_x = x0
    # output_x = x0

    # Figure out the correct way to do this!
    k_length = order + 2
    {length_x} = Nx.shape(x0)
    # , type: Nx.type(x0))
    k_vals = Nx.broadcast(0.0, {length_x, k_length})
    temp = %{temp | k_vals: k_vals}

    step_acc = %StepAccumulator{
      t_new: t_start,
      x_new: x0,
      dt: dt,
      k_vals: k_vals
    }

    step_forward(step_acc, t_start, t_end, stepper_fn, interpolate_fn, ode_fn)
    %__MODULE__{temp: temp}
  end

  def step_forward(step_acc, t_old, t_end, _stepper_fn, _interpolate_fn, _ode_fn) when t_old >= t_end do
    step_acc
  end

  def step_forward(step_acc, _t_old, t_end, stepper_fn, interpolate_fn, ode_fn) do
    {step_acc, error} = compute_step(step_acc, stepper_fn, ode_fn)

    step_acc =
      if Nx.less(error, 1.0) == @nx_true do
        step_acc = %{
          step_acc
          | count_loop: step_acc.count_loop + 1,
            i_step: step_acc.i_step + 1,
            i_reject: 0,
            terminal_event: false,
            terminal_output: false
        }

        step_acc = %{step_acc | ode_t: [step_acc.t_new | step_acc.ode_t]}
        step_acc = %{step_acc | ode_x: [step_acc.x_new | step_acc.ode_x]}

        IO.inspect(step_acc.t_old, label: "step_acc.t_old")
        IO.inspect(step_acc.t_new, label: "step_acc.t_new")
        IO.inspect(@refine + 1, label: "refine + 1")
        IO.inspect(Nx.type(step_acc.x_old), label: "type of x_old")

        tadd = Nx.linspace(step_acc.t_old, step_acc.t_new, n: @refine + 1, type: Nx.type(step_acc.x_old))
        IO.puts("-----------------------------------")
        IO.inspect(tadd, label: "tadd")

        # x_out = interpolate_fn.(t, x, der, t_out)

        step_acc
      else
        step_acc
      end

    step_forward(step_acc, Nx.to_number(step_acc.t_new), t_end, stepper_fn, interpolate_fn, ode_fn)
  end

  def compute_step(step_acc, stepper_fn, ode_fn) do
    x_old = step_acc.x_new
    t_old = step_acc.t_new
    options_comp_old = step_acc.options_comp
    k_vals = step_acc.k_vals
    dt = step_acc.dt

    {_t_new, options_comp} = kahan_sum(t_old, options_comp_old, dt)
    {t_next, x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals)

    # Pass these in as options:
    abs_tol = 1.0e-06
    rel_tol = 1.0e-03
    norm_control = false
    error = Utils.abs_rel_norm(x_next, x_old, x_est, abs_tol, rel_tol, norm_control: norm_control)

    {%{
       step_acc
       | count_cycles: step_acc.count_cycles + 1,
         x_old: x_old,
         t_old: t_old,
         x_new: x_next,
         t_new: t_next,
         k_vals: k_vals,
         options_comp: options_comp
     }, error}
  end

  @doc """
  Implements the Kahan summation algorithm, also known as compensated summation.
  Based on this [code in Octave](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/kahan.m).

  The algorithm significantly reduces the numerical error in the total
  obtained by adding a sequence of finite precision floating point numbers
  compared to the straightforward approach.  For more details
  see [this Wikipedia entry](http://en.wikipedia.org/wiki/Kahan_summation_algorithm).
  This function is called by AdaptiveStepsize.integrate to better catch
  equality comparisons.

  The first input argument is the variable that will contain the summation.
  This variable is also returned as the first output argument in order to
  reuse it in subsequent calls to `Integrator.AdaptiveStepsize.kahan_sum/3` function.

  The second input argument contains the compensation term and is returned
  as the second output argument so that it can be reused in future calls of
  the same summation.

  The third input argument `term` is the variable to be added to `sum`.
  """
  defn kahan_sum(sum, comp, term) do
    # Octave code:
    # y = term - comp;
    # t = sum + y;
    # comp = (t - sum) - y;
    # sum = t;

    y = term - comp
    t = sum + y
    comp = t - sum - y
    sum = t

    {sum, comp}
  end

  def ode_event_handler() do
  end
end
