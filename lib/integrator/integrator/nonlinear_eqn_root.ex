defmodule Integrator.NonlinearEqnRoot do
  @moduledoc false

  import Integrator.Utils, only: [sign: 1, epsilon: 1, unique: 1]

  alias Integrator.NonlinearEqnRoot.BracketingFailureError
  alias Integrator.Utils

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
    # x (and fx) are the actual found values (i.e., fx should be very close to zero):
    :x,
    :fx,
    #
    :mu_ba,
    #
    fn_eval_count: 0,
    iteration_count: 0,
    # Change itype to a more descriptive atom later:
    itype: 1,
    #
    bracket_t: [],
    bracket_y: []
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

  def set_machine_eps([{:machine_eps, machine_eps}] = opts) when not is_nil(machine_eps), do: opts
  def set_machine_eps(opts), do: opts |> Keyword.merge(machine_eps: epsilon(opts[:type]))

  def find_zero(zero_fn, a, b, opts \\ []) do
    # Do some tests for opts
    opts = default_opts() |> Keyword.merge(opts) |> set_tolerance() |> set_machine_eps()

    fa = zero_fn.(a)
    fb = zero_fn.(b)
    {u, fu} = if abs(fa) < abs(fb), do: {a, fa}, else: {b, fb}

    # Write a test for this swap:
    {a, b, fa, fb} = if b < a, do: {b, a, fb, fa}, else: {a, b, fa, fb}

    mu = 0.5
    mu_ba = mu * (b - a)

    z = %__MODULE__{
      a: a,
      b: b,
      d: u,
      e: u,
      u: u,
      #
      fa: fa,
      fb: fb,
      fd: fu,
      fe: fu,
      fu: fu,
      #
      fn_eval_count: 2,
      itype: 1,
      mu_ba: mu_ba
    }

    # Write a test for this:
    if sign(z.fa) * sign(z.fb) > 0.0 do
      raise "fzero: not an valid initial bracket"
    end

    case converged?(z, opts[:machine_eps], opts[:tolerance]) do
      :continue -> iterate(z, :continue, zero_fn, opts)
      :halt -> %{z | x: u, fx: fu}
    end
  end

  def iterate(z, :halt, _zero_fn, _opts) do
    %{z | x: z.u, fx: z.fu}
  end

  def iterate(z, _status, zero_fn, opts) do
    machine_eps = opts[:machine_eps]
    tolerance = opts[:tolerance]

    {status1, z} =
      z
      |> zcase()
      |> c_too_close_to_a_or_b?(machine_eps, tolerance)
      |> compute_new_point(zero_fn)
      |> check_for_non_monotonicity()
      |> bracket()

    # call output function here!

    z =
      z
      |> skip_bisection_if_successful_reduction()
      |> update_u()

    status2 = converged?(z, machine_eps, tolerance)

    iterate(z, status(status1, status2), zero_fn, opts)
  end

  def status(:halt, _), do: :halt
  def status(_, :halt), do: :halt
  def status(_, _), do: :continue

  def zcase(%{itype: 1} = z) do
    #   if (abs (fa) <= 1e3*abs (fb) && abs (fb) <= 1e3*abs (fa))
    #   # Secant step.
    #   c = u - (a - b) / (fa - fb) * fu;
    # else
    #   # Bisection step.
    #   c = 0.5*(a + b);
    # endif
    # d = u; fd = fu;
    # itype = 5;

    c =
      if abs(z.fa) <= 1000 * abs(z.fb) && abs(z.fb) <= 1000 * abs(z.fa) do
        interpolate(z, :secant)
      else
        interpolate(z, :bisect)
      end

    %{z | c: c, d: z.u, fd: z.fu, itype: 5}
  end

  def zcase(%{itype: 2} = z) do
    zcase_two_and_three(z)
  end

  def zcase(%{itype: 3} = z) do
    zcase_two_and_three(z)
  end

  def zcase(%{itype: 4} = z) do
    # Octave:
    #   # Double secant step.
    #   c = u - 2*(b - a)/(fb - fa)*fu;
    #   # Bisect if too far.
    #   if (abs (c - u) > 0.5*(b - a))
    #     c = 0.5 * (b + a);
    #   endif
    #   itype = 5;

    c = interpolate(z, :double_secant)

    c =
      if abs(c - z.u) > 0.5 * (z.b - z.a) do
        # Bisect if too far:
        interpolate(z, :bisect)
      else
        c
      end

    %{z | itype: 5, c: c}
  end

  def zcase(%{itype: 5} = z) do
    # Octave:
    #   # Bisection step.
    #   c = 0.5 * (b + a);
    #   itype = 2;
    c = interpolate(z, :bisect)
    %{z | itype: 2, c: c}
  end

  def zcase_two_and_three(z) do
    length = length(Utils.unique([z.fa, z.fb, z.fd, z.fe]))

    c =
      case length do
        4 ->
          interpolate(z, :inverse_cubic_interpolation)

        _ ->
          if sign(z.c - z.a) * sign(z.c - z.b) > 0 do
            interpolate(z, :quadratic_interpolation_plus_newton)
          else
            # what do we do here?  it's not handled in fzero.m
            z.c
          end
      end

    %{z | itype: z.itype + 1, c: c}
  end

  def interpolate(z, :quadratic_interpolation_plus_newton) do
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
          # Octave does a break here - is the c = 0 caught downstream? Need to handle this case somehow"
          # Note that there is NO test case for this case, as I couldn't figure out how to set up
          # the initial conditions to reach here
          z.a - a0 / a1
        else
          c - pc / pdc
        end
      end)
    else
      c
    end
  end

  def interpolate(z, :inverse_cubic_interpolation) do
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

  def interpolate(z, :double_secant) do
    z.u - 2.0 * (z.b - z.a) / (z.fb - z.fa) * z.fu
  end

  def interpolate(z, :bisect) do
    0.5 * (z.b + z.a)
  end

  def interpolate(z, :secant) do
    z.u - (z.a - z.b) / (z.fa - z.fb) * z.fu
  end

  def too_far?(z) do
    abs(z.c - z.u) > 0.5 * (z.b - z.a)
  end

  def next_compute(:bisect), do: :quadratic

  def compute_new_point(z, zero_fn) do
    fc = zero_fn.(z.c)
    #  fval = fc    What is this used for?
    # Move the incrementing of the interation count elsewhere?
    %{z | fc: fc, x: z.c, fx: fc, fn_eval_count: z.fn_eval_count + 1, iteration_count: z.iteration_count + 1}
  end

  # Modification 2: skip inverse cubic interpolation nonmonotonicity is detected.
  def check_for_non_monotonicity(z) do
    if sign(z.fc - z.fa) * sign(z.fc - z.fb) >= 0 do
      # The new point broke monotonicity.
      # Disable inverse cubic:
      %{z | fe: z.fc}
    else
      %{z | e: z.d, fe: z.fd}
    end
  end

  def c_too_close_to_a_or_b?(z, machine_eps, tolerance) do
    delta = 2 * 0.7 * (2 * abs(z.u) * machine_eps + tolerance)

    c =
      if z.b - z.a <= 2 * delta do
        (z.a + z.b) / 2
      else
        max(z.a + delta, min(z.b - delta, z.c))
      end

    %{z | c: c}
  end

  def bracket(z) do
    if sign(z.fa) * sign(z.fc) < 0 do
      {:continue, %{z | d: z.b, fd: z.fb, b: z.c, fb: z.fc}}
    else
      if sign(z.fb) * sign(z.fc) < 0 do
        {:continue, %{z | d: z.a, fd: z.fa, a: z.c, fa: z.fc}}
      else
        if z.fc == 0.0 do
          {:halt, %{z | a: z.c, b: z.c, fa: z.fc, fb: z.fc}}
        else
          raise BracketingFailureError, step: z
        end
      end
    end
  end

  def update_u(z) do
    # Octave:
    #   if (abs (fa) < abs (fb))
    #     u = a; fu = fa;
    #   else
    #     u = b; fu = fb;
    #   end

    if abs(z.fa) < abs(z.fb) do
      %{z | u: z.a, fu: z.fa}
    else
      %{z | u: z.b, fu: z.fb}
    end
  end

  def converged?(z, machine_eps, tolerance) do
    if z.b - z.a <= 2 * (2 * abs(z.u) * machine_eps + tolerance) do
      :halt
    else
      :continue
    end
  end

  def skip_bisection_if_successful_reduction(z) do
    # Octave:
    #   if (itype == 5 && (b - a) <= mba)
    #     itype = 2;
    #   endif
    #   if (itype == 2)
    #     mba = mu * (b - a);
    #   endif

    z =
      if z.itype == 5 && z.b - z.a <= z.mu_ba do
        %{z | itype: 2}
      else
        z
      end

    if z.itype == 2 do
      %{z | mu_ba: z.mu * (z.b - z.a)}
    else
      z
    end
  end
end
