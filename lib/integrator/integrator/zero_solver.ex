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

  def compute(z, :quadratic_interpolation_plus_newton) do
    a0 = z.fa
    a1 = (z.fb - z.fa) / (z.b - z.a)
    a2 = ((z.fd - z.fb) / (z.d - z.b) - a1) / (z.d - z.a)

    ## Modification 1: this is simpler and does not seem to be worse.
    c = z.a - a0 / a1

    if a2 != 0 do
      1..z.itype
      |> Enum.reduce(c, fn _i, c ->
        pc = a0 + (a1 + a2 * (c - z.b)) * (c - z.a)
        pdc = a1 + a2 * (2 * c - z.a - z.b)

        if pdc == 0 do
          _c = z.a - a0 / a1
          raise "Need to handle this case somehow"
        end

        c - pc / pdc
      end)
    else
      c
    end
  end

  def compute(z, :inverse_cubic_interpolation) do
    q11 = (z.d - z.e) * z.fd / (z.fe - z.fd)
    q21 = (z.b - z.d) * z.fb / (z.fd - z.fb)
    q31 = (z.a - z.b) * z.fa / (z.fb - z.fa)
    d21 = (z.b - z.d) * z.fd / (z.fd - z.fb)
    d31 = (z.a - z.b) * z.fb / (z.fb - z.fa)

    q22 = (d21 - q11) * z.fb / (z.fe - z.fb)
    q32 = (d31 - q21) * z.fa / (z.fd - z.fa)
    d32 = (d31 - q21) * z.fd / (z.fd - z.fa)
    q33 = (d32 - q22) * z.fa / (z.fe - z.fa)

    z.a + q31 + q32 + q33
  end

  def compute(z, :double_secant) do
    z.u - 2.0 * (z.b - z.a) / (z.fb - z.fa) * z.fu
  end

  def compute(z, :bisect) do
    0.5 * (z.b + z.a)
  end

  def compute(z, :secant) do
    z.u - (z.a - z.b) / (z.fa - z.fb) * z.fu
  end

  def too_far?(z) do
    abs(z.c - z.u) > 0.5 * (z.b - z.a)
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
