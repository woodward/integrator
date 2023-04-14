defmodule Integrator.AdaptiveStepsize do
  @moduledoc false
  import Nx.Defn
  alias Integrator.Utils

  defmodule ComputedStep do
    @moduledoc false
    defstruct [
      :t_new,
      :x_new,
      :k_vals,
      :options_comp
    ]
  end

  defstruct [
    :t_old,
    :t_new,
    :x_old,
    :x_new,
    :dt,
    :k_vals,
    #
    options_comp: 0.0,
    #
    count_loop__increment_step: 0,
    count_cycles__compute_step: 0,
    count_save: 2,
    #
    i_reject: 0,
    i_step: 0,
    #
    unhandled_termination: true,
    terminal_event: false,
    terminal_output: false,
    #
    ode_t: [],
    ode_x: [],
    output_t: [],
    output_x: []
  ]

  @stepsize_factor_min 0.8
  @stepsize_factor_max 1.5

  @default_refine 4
  @default_max_number_of_errors 5_000
  @default_max_step 2.0

  @epsilon 2.2204e-16

  @nx_true Nx.tensor(1, type: :u8)
  # @nx_false Nx.tensor(0, type: :u8)

  def default_opts() do
    [
      epsilon: @epsilon,
      max_number_of_errors: @default_max_number_of_errors,
      max_step: @default_max_step,
      refine: @default_refine
    ]
  end

  @doc """

  See [Wikipedia](https://en.wikipedia.org/wiki/Adaptive_stepsize)
  """
  def integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts \\ []) do
    opts = default_opts() |> Keyword.merge(Utils.default_opts()) |> Keyword.merge(opts)

    step = %__MODULE__{
      t_new: t_start,
      x_new: x0,
      dt: initial_tstep,
      k_vals: initial_empty_k_vals(order, x0),
      output_t: [t_start],
      output_x: [x0],
      ode_t: [t_start],
      ode_x: [x0]
    }

    step_forward(step, t_start, t_end, stepper_fn, interpolate_fn, ode_fn, order, opts)
    |> reverse_results()
  end

  def initial_empty_k_vals(order, x) do
    # Figure out the correct way to do this!
    k_length = order + 2

    {length_x} = Nx.shape(x)
    Nx.broadcast(0.0, {length_x, k_length})
  end

  def step_forward(step, t_old, t_end, _stepper_fn, _interpolate_fn, _ode_fn, _order, _opts) when t_old >= t_end do
    step
  end

  def step_forward(step, t_old, t_end, stepper_fn, interpolate_fn, ode_fn, order, opts) do
    {new_step, error_est} = compute_step(step, stepper_fn, ode_fn, opts)
    step = step |> increment_compute_counter()

    step =
      if Nx.less(error_est, 1.0) == @nx_true do
        step
        |> increment_step(new_step)
        |> interpolate(interpolate_fn, opts[:refine])

        # call to output function
      else
        # Error condition
        step = %{step | i_reject: step.i_reject + 1}

        if step.i_reject > opts[:max_number_of_errors] do
          raise "Too many errors"
        end

        step
      end

    dt = compute_next_timestep(step.dt, Nx.to_number(error_est), order, t_old, t_end, opts)
    step = %{step | dt: dt}
    t_old = if step.i_reject > 0, do: Nx.to_number(step.t_old) + dt, else: Nx.to_number(step.t_new)

    step
    |> step_forward(t_old, t_end, stepper_fn, interpolate_fn, ode_fn, order, opts)
  end

  def reverse_results(step) do
    %{
      step
      | output_x: step.output_x |> Enum.reverse(),
        output_t: step.output_t |> Enum.reverse(),
        #
        ode_x: step.ode_x |> Enum.reverse(),
        ode_t: step.ode_t |> Enum.reverse()
    }
  end

  @doc """
  Formula taken from Hairer
  """
  def compute_next_timestep(dt, error, order, t_old, t_end, opts) do
    # Avoid divisions by zero:
    error = error + opts[:epsilon]

    # factor should be cached somehow; perhaps passed in in the options?
    factor = Math.pow(0.38, 1.0 / (order + 1))

    foo = factor * Math.pow(1 / error, 1 / (order + 1))

    dt = dt * min(@stepsize_factor_max, max(@stepsize_factor_min, foo))
    dt = min(abs(dt), opts[:max_step])

    # ## Make sure we don't go past t_end:
    min(abs(dt), abs(t_end - t_old))
  end

  def increment_step(step, computed_step) do
    %{
      step
      | count_loop__increment_step: step.count_loop__increment_step + 1,
        i_step: step.i_step + 1,
        i_reject: 0,
        terminal_event: false,
        terminal_output: false,
        ode_t: [step.t_new | step.ode_t],
        ode_x: [step.x_new | step.ode_x],
        #
        x_old: step.x_new,
        t_old: step.t_new,
        x_new: computed_step.x_new,
        t_new: computed_step.t_new,
        k_vals: computed_step.k_vals,
        options_comp: computed_step.options_comp
    }
  end

  defp increment_compute_counter(step) do
    %{step | count_cycles__compute_step: step.count_cycles__compute_step + 1}
  end

  def compute_step(step, stepper_fn, ode_fn, opts) do
    x_old = step.x_new
    t_old = step.t_new
    options_comp_old = step.options_comp
    k_vals = step.k_vals
    dt = step.dt

    {_t_new, options_comp} = kahan_sum(t_old, options_comp_old, dt)
    {t_next, x_next, x_est, k_vals} = stepper_fn.(ode_fn, t_old, x_old, dt, k_vals)

    # Pass these in as options:
    norm_control = false
    error = Utils.abs_rel_norm(x_next, x_old, x_est, opts[:abs_tol], opts[:rel_tol], norm_control: norm_control)

    {%ComputedStep{
       x_new: x_next,
       t_new: t_next,
       k_vals: k_vals,
       options_comp: options_comp
     }, error}
  end

  def interpolate(step, _interpolate_fn, refine) when refine == 1 do
    step
  end

  def interpolate(step, interpolate_fn, refine) do
    tadd = Nx.linspace(step.t_old, step.t_new, n: refine + 1, type: Nx.type(step.x_old))
    # Get rid of the first element (tadd[0]) via this slice:
    tadd = Nx.slice_along_axis(tadd, 1, refine, axis: 0)

    t = Nx.stack([step.t_old, step.t_new])
    x = Nx.stack([step.x_old, step.x_new]) |> Nx.transpose()

    x_out = interpolate_fn.(t, x, step.k_vals, tadd)
    x_out_as_cols = Utils.columns_as_list(x_out, 0, refine - 1) |> Enum.reverse()

    step = %{step | output_x: x_out_as_cols ++ step.output_x}
    %{step | output_t: (Nx.to_list(tadd) |> Enum.reverse()) ++ step.output_t}
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
