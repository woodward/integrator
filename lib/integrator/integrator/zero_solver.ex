defmodule Integrator.ZeroSolver do
  @moduledoc false

  import Integrator.Utils, only: [sign: 1, epsilon: 1, unique: 1]

  defstruct [
    :a,
    :b,
    :c,
    :d,
    :e,
    :u,
    #
    :fa,
    :fb,
    :fc,
    :fd,
    :fe,
    :fu,
    #
    :mu_ba,
    #
    fn_eval_count: 0,
    iteration_count: 0,
    # Change itype to a more description atom later:
    # Maybe these don't need to be stored on the struct, but will be just what is invoked?
    itype: 1,
    type: :foo,
    next_type: :bar
  ]

  @default_max_fn_eval_count 1000
  @default_max_iterations 1000
  @default_type :f64
  @default_tolerance 0.01

  def default_opts do
    [
      max_iterations: @default_max_iterations,
      max_fn_eval_count: @default_max_fn_eval_count,
      type: @default_type
    ]
  end

  def set_tolerance([{:tolerance, tolerance}] = opts) when not is_nil(tolerance), do: opts
  def set_tolerance(opts), do: opts |> Keyword.merge(tolerance: epsilon(opts[:type]))

  def find(zero_fn, a, b, opts \\ []) do
    # Do some tests for opts
    opts = default_opts |> Keyword.merge(opts) |> set_tolerance()

    fa = zero_fn.(a)
    fb = zero_fn.(b)

    # Write a test for this swap:
    {a, b, fa, fb} = if b < a, do: {b, a, fb, fa}, else: {a, b, fa, fb}

    z = %__MODULE__{a: a, b: b, fa: fa, fb: fb, fn_eval_count: 2}

    # Write a test for this:
    if sign(z.fa) * sign(z.fb) > 0.0 do
      raise "fzero: not an valid initial bracketing"
    end

    mu = 0.5

    z
  end

  def compute(:inverse_cubic_interpolation) do
  end

  def compute(:quadratic_interpolation_plus_newton) do
  end

  def compute(:secant) do
  end

  def compute(:double_secant) do
  end

  def compute(:bisection) do
  end

  def next_compute(:bisect), do: :quadratic

  def compute_new_point(z) do
    z
  end

  def converged?(z, machine_eps, tolerance) do
    if z.b - z.a <= 2 * (2 * abs(z.u) * machine_eps + tolerance) do
      :halt
    else
      :continue
    end
  end
end
