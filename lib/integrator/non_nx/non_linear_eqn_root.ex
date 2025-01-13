defmodule Integrator.NonNx.NonLinearEqnRoot do
  @moduledoc """
  Finds the roots (i.e., zeros) of a non-linear equation.
  Based on [`fzero.m`](https://github.com/gnu-octave/octave/blob/default/scripts/optimization/fzero.m)
  from Octave.

  This is essentially the ACM algorithm 748: Enclosing Zeros of Continuous Functions
  due to Alefeld, Potra and Shi, ACM Transactions on Mathematical Software, Vol. 21,
  No. 3, September 1995. Although the workflow is the same, the structure of
  the algorithm has been transformed non-trivially; also, the algorithm has also been
  slightly modified.
  """

  alias Integrator.NonNx.NonLinearEqnRoot.BracketingFailureError
  alias Integrator.NonNx.NonLinearEqnRoot.InvalidInitialBracketError
  alias Integrator.NonNx.NonLinearEqnRoot.MaxFnEvalsExceededError
  alias Integrator.NonNx.NonLinearEqnRoot.MaxIterationsExceededError

  @type zero_fn_t :: (float() -> float())
  @type output_fn_t :: (float(), float() -> any())

  @type interpolation_type ::
          :bisect
          | :double_secant
          | :inverse_cubic_interpolation
          | :quadratic_interpolation_plus_newton
          | :secant

  @type convergence_status :: :halt | :continue

  @type iter_type :: 1 | 2 | 3 | 4 | 5

  @type t :: %__MODULE__{
          a: float() | nil,
          b: float() | nil,
          c: float() | nil,
          d: float() | nil,
          e: float() | nil,
          u: float() | nil,
          #
          # Function evaluations; e.g., fb is fn(b):
          fa: float() | nil,
          fb: float() | nil,
          fc: float() | nil,
          fd: float() | nil,
          fe: float() | nil,
          fu: float() | nil,
          #
          # x (and fx) are the actual found values (i.e., fx should be very close to zero):
          x: float() | nil,
          fx: float() | nil,
          #
          mu_ba: float() | nil,
          #
          fn_eval_count: integer(),
          iteration_count: integer(),
          # Change iter_type to a more descriptive atom later (possibly?):
          iter_type: iter_type()
        }

  defstruct [
    :a,
    :b,
    :c,
    :d,
    :e,
    :u,
    #
    # Function evaluations; e.g., fb is fn(b):
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
    # Change iter_type to a more descriptive atom later (possibly?):
    iter_type: 1
  ]

  options = [
    max_iterations: [
      type: :integer,
      doc: "The maximum allowed number of iterations when finding a root.",
      default: 1000
    ],
    max_fn_eval_count: [
      type: :integer,
      doc: "The maximum allowed number of function evaluations when finding a root.",
      default: 1000
    ],
    type: [
      type: {:in, [:f32, :f64]},
      doc: "The Nx type.",
      default: :f64
    ],
    machine_eps: [
      type: :float,
      doc: "The machine epsilon. Defaults to Nx.constants.epsilon/1 for this Nx type."
    ],
    tolerance: [
      type: :float,
      doc: "The tolerance for the convergence when finding a root. Defaults to Nx.Constants.epsilon/1 for this Nx type."
    ],
    nonlinear_eqn_root_output_fn: [
      # Ideally the type for this should be set to a function with arity 2, but I could not get that to work:
      type: :any,
      doc: "An output function to call so intermediate results can be retrieved when finding a root.",
      default: nil
    ]
  ]

  @options_schema NimbleOptions.new!(options)
  @type options_t() :: unquote(NimbleOptions.option_typespec(@options_schema))

  def options_schema, do: @options_schema

  @initial_mu 0.5

  @doc """
  Finds a zero for a function in an interval `[a, b]` (if the 2nd argument is a list) or
  in the vicinity of `a` (if the 2nd argument is a float).

  ## Options

  #{NimbleOptions.docs(@options_schema)}

  """

  # I can't get this to work for some unknown reason:
  # @spec find_zero(zero_fn_t(), [float()] | float(), options_t()) :: t()

  @spec find_zero(zero_fn_t(), [float()] | float(), Keyword.t(), integer()) :: t()
  def find_zero(zero_fn, initial_values, opts \\ [], fn_evals \\ 0)

  def find_zero(zero_fn, [a, b], opts, fn_evals) do
    opts = opts |> NimbleOptions.validate!(@options_schema) |> merge_default_opts()

    fa = zero_fn.(a)
    fb = zero_fn.(b)
    fn_eval_count = 2 + fn_evals
    {u, fu} = if abs(fa) < abs(fb), do: {a, fa}, else: {b, fb}
    {a, b, fa, fb} = if b < a, do: {b, a, fb, fa}, else: {a, b, fa, fb}

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
      fn_eval_count: fn_eval_count,
      iter_type: 1,
      mu_ba: @initial_mu * (b - a)
    }

    if sign(z.fa) * sign(z.fb) > 0.0, do: raise(InvalidInitialBracketError, step: z)

    case converged?(z, opts[:machine_eps], opts[:tolerance]) do
      :continue -> iterate(z, :continue, zero_fn, opts)
      :halt -> %{z | x: u, fx: fu}
    end
  end

  def find_zero(zero_fn, solo_point, opts, _fn_evals) do
    second_point = find_2nd_starting_point(zero_fn, solo_point)

    find_zero(zero_fn, [solo_point, second_point.b], opts, second_point.fn_eval_count)
  end

  @spec bracket_x(t()) :: [float()]
  def bracket_x(z) do
    [z.a, z.b]
  end

  @spec bracket_fx(t()) :: [float()]
  def bracket_fx(z) do
    [z.fa, z.fb]
  end

  def option_keys, do: NimbleOptions.validate!([], @options_schema) |> Keyword.keys()

  # ===========================================================================
  # Private functions below here:

  @spec iterate(t(), atom(), zero_fn_t(), Keyword.t()) :: t()
  defp iterate(z, :halt, _zero_fn, _opts), do: z

  defp iterate(z, _status, zero_fn, opts) do
    machine_eps = opts[:machine_eps]
    tolerance = opts[:tolerance]

    {status_1, z} =
      z
      |> compute_iteration()
      |> adjust_if_too_close_to_a_or_b(machine_eps, tolerance)
      |> fn_eval_new_point(zero_fn, opts)
      |> check_for_non_monotonicity()
      |> bracket()

    z =
      z
      |> skip_bisection_if_successful_reduction()
      |> update_u()
      |> call_output_fn(opts[:nonlinear_eqn_root_output_fn])

    status_2 = converged?(z, machine_eps, tolerance)

    iterate(z, halt?(status_1, status_2), zero_fn, opts)
  end

  @spec halt?(convergence_status(), convergence_status()) :: convergence_status()
  defp halt?(:halt, _), do: :halt
  defp halt?(_, :halt), do: :halt
  defp halt?(_, _), do: :continue

  @spec compute_iteration(t()) :: t()
  defp compute_iteration(%{iter_type: 1} = z) do
    # Octave:
    #   if (abs (fa) <= 1e3*abs (fb) && abs (fb) <= 1e3*abs (fa))
    #     # Secant step.
    #     c = u - (a - b) / (fa - fb) * fu;
    #   else
    #     # Bisection step.
    #     c = 0.5*(a + b);
    #   endif
    #   d = u; fd = fu;
    #   iter_type = 5;

    # What is the significance or meaning of the 1000 here? Replace with a more descriptive module variable
    c =
      if abs(z.fa) <= 1000 * abs(z.fb) && abs(z.fb) <= 1000 * abs(z.fa) do
        interpolate(z, :secant)
      else
        interpolate(z, :bisect)
      end

    %{z | c: c, d: z.u, fd: z.fu, iter_type: 5}
  end

  defp compute_iteration(%{iter_type: 2} = z) do
    compute_iteration_two_or_three(z)
  end

  defp compute_iteration(%{iter_type: 3} = z) do
    compute_iteration_two_or_three(z)
  end

  defp compute_iteration(%{iter_type: 4} = z) do
    # Octave:
    #   # Double secant step.
    #   c = u - 2*(b - a)/(fb - fa)*fu;
    #   # Bisect if too far.
    #   if (abs (c - u) > 0.5*(b - a))
    #     c = 0.5 * (b + a);
    #   endif
    #   iter_type = 5;

    c = interpolate(z, :double_secant)

    c =
      if too_far?(c, z) do
        # Bisect if too far:
        interpolate(z, :bisect)
      else
        c
      end

    %{z | iter_type: 5, c: c}
  end

  defp compute_iteration(%{iter_type: 5} = z) do
    # Octave:
    #   # Bisection step.
    #   c = 0.5 * (b + a);
    #   iter_type = 2;
    c = interpolate(z, :bisect)
    %{z | iter_type: 2, c: c}
  end

  @spec compute_iteration_two_or_three(t()) :: t()
  defp compute_iteration_two_or_three(z) do
    c =
      case length(unique([z.fa, z.fb, z.fd, z.fe])) do
        4 ->
          interpolate(z, :inverse_cubic_interpolation)

        length ->
          if length < 4 || sign(z.c - z.a) * sign(z.c - z.b) > 0 do
            interpolate(z, :quadratic_interpolation_plus_newton)
          else
            # what do we do here?  it's not handled in fzero.m...
            z.c
          end
      end

    %{z | iter_type: z.iter_type + 1, c: c}
  end

  @search_values [-0.01, 0.025, -0.05, 0.10, -0.25, 0.50, -1.0, 2.5, -5.0, 10.0, -50.0, 100.0, 500.0, 1000.0]

  defmodule SearchFor2ndPoint do
    @moduledoc false
    defstruct [:a, :fa, :b, :fb, :fn_eval_count]
  end

  @type search_for_2nd_point_t :: %SearchFor2ndPoint{
          a: float() | nil,
          b: float() | nil,
          #
          # Function evaluations; e.g., fb is fn(b):
          fa: float() | nil,
          fb: float() | nil,
          #
          fn_eval_count: integer()
        }

  @spec find_2nd_starting_point(zero_fn_t(), float()) :: map()
  defp find_2nd_starting_point(zero_fn, a) do
    # For very small values, switch to absolute rather than relative search:
    a =
      if abs(a) < 0.001 do
        if a == 0, do: 0.1, else: sign(a) * 0.1
      else
        a
      end

    fa = zero_fn.(a)
    x = %SearchFor2ndPoint{a: a, fa: fa, b: nil, fb: nil, fn_eval_count: 1}

    # Search in an ever-widening range around the initial point:
    searching_for_2nd_point(:continue, zero_fn, x, @search_values)
  end

  @spec searching_for_2nd_point(atom(), zero_fn_t(), search_for_2nd_point_t(), [float()]) :: map()
  defp searching_for_2nd_point(:found, _zero_fn, x, _search_values), do: x

  defp searching_for_2nd_point(:continue, zero_fn, x, search_values) do
    [search | rest_of_search_values] = search_values
    b = x.a + x.a * search
    fb = zero_fn.(b)
    x = %{x | b: b, fb: fb, fn_eval_count: x.fn_eval_count + 1}
    status = if sign(x.fa) * sign(fb) <= 0, do: :found, else: :continue
    searching_for_2nd_point(status, zero_fn, x, rest_of_search_values)
  end

  @spec interpolate(t(), interpolation_type()) :: float()
  defp interpolate(z, :quadratic_interpolation_plus_newton) do
    a0 = z.fa
    a1 = (z.fb - z.fa) / (z.b - z.a)
    a2 = ((z.fd - z.fb) / (z.d - z.b) - a1) / (z.d - z.a)

    ## Modification 1: this is simpler and does not seem to be worse.
    c = z.a - a0 / a1

    if a2 != 0 do
      1..z.iter_type
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

  defp interpolate(z, :inverse_cubic_interpolation) do
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

  defp interpolate(z, :double_secant) do
    z.u - 2.0 * (z.b - z.a) / (z.fb - z.fa) * z.fu
  end

  defp interpolate(z, :bisect) do
    0.5 * (z.b + z.a)
  end

  defp interpolate(z, :secant) do
    z.u - (z.a - z.b) / (z.fa - z.fb) * z.fu
  end

  @spec too_far?(float(), t()) :: boolean()
  defp too_far?(c, z) do
    abs(c - z.u) > 0.5 * (z.b - z.a)
  end

  @spec fn_eval_new_point(t(), zero_fn_t(), Keyword.t()) :: t()
  defp fn_eval_new_point(z, zero_fn, opts) do
    fc = zero_fn.(z.c)
    # Perhaps move the incrementing of the iteration count elsewhere?
    iteration_count = z.iteration_count + 1
    fn_eval_count = z.fn_eval_count + 1

    if iteration_count > opts[:max_iterations] do
      raise MaxIterationsExceededError, step: z, iteration_count: iteration_count
    end

    if fn_eval_count > opts[:max_fn_eval_count] do
      raise MaxFnEvalsExceededError, step: z, fn_eval_count: fn_eval_count
    end

    %{
      z
      | fc: fc,
        x: z.c,
        fx: fc,
        fn_eval_count: fn_eval_count,
        iteration_count: iteration_count
    }
  end

  # Modification 2: skip inverse cubic interpolation if nonmonotonicity is detected
  @spec check_for_non_monotonicity(t()) :: t()
  defp check_for_non_monotonicity(z) do
    if sign(z.fc - z.fa) * sign(z.fc - z.fb) >= 0 do
      # The new point broke monotonicity.
      # Disable inverse cubic:
      %{z | fe: z.fc}
    else
      %{z | e: z.d, fe: z.fd}
    end
  end

  @spec adjust_if_too_close_to_a_or_b(t(), float(), float()) :: t()
  defp adjust_if_too_close_to_a_or_b(z, machine_eps, tolerance) do
    delta = 2 * 0.7 * (2 * abs(z.u) * machine_eps + tolerance)

    c =
      if z.b - z.a <= 2 * delta do
        (z.a + z.b) / 2
      else
        max(z.a + delta, min(z.b - delta, z.c))
      end

    %{z | c: c}
  end

  @spec bracket(t()) :: {convergence_status(), t()}
  defp bracket(z) do
    {status, z} =
      if sign(z.fa) * sign(z.fc) < 0 do
        {:continue, %{z | d: z.b, fd: z.fb, b: z.c, fb: z.fc}}
      else
        if sign(z.fb) * sign(z.fc) < 0 do
          {:continue, %{z | d: z.a, fd: z.fa, a: z.c, fa: z.fc}}
        else
          if z.fc == 0.0 do
            {:halt, %{z | a: z.c, b: z.c, fa: z.fc, fb: z.fc}}
          else
            # Should never reach here
            raise BracketingFailureError, step: z
          end
        end
      end

    {status, z}
  end

  @spec call_output_fn(t(), output_fn_t()) :: t()
  defp call_output_fn(z, nil = _output_fn), do: z

  defp call_output_fn(z, output_fn) do
    output_fn.(z.x, z)
    z
  end

  @spec update_u(t()) :: t()
  defp update_u(z) do
    # Octave:
    #   if (abs (fa) < abs (fb))
    #     u = a; fu = fa;
    #   else
    #     u = b; fu = fb;
    #   endif

    if abs(z.fa) < abs(z.fb) do
      %{z | u: z.a, fu: z.fa}
    else
      %{z | u: z.b, fu: z.fb}
    end
  end

  @spec converged?(t(), float(), float()) :: convergence_status()
  defp converged?(z, machine_eps, tolerance) do
    if z.b - z.a <= 2 * (2 * abs(z.u) * machine_eps + tolerance) do
      :halt
    else
      :continue
    end
  end

  @spec skip_bisection_if_successful_reduction(t()) :: t()
  defp skip_bisection_if_successful_reduction(z) do
    # Octave:
    #   if (iter_type == 5 && (b - a) <= mba)
    #     iter_type = 2;
    #   endif
    #   if (iter_type == 2)
    #     mba = mu * (b - a);
    #   endif

    z =
      if z.iter_type == 5 && z.b - z.a <= z.mu_ba do
        %{z | iter_type: 2}
      else
        z
      end

    if z.iter_type == 2 do
      # Should this really be @initial_mu here?  or should it be mu_ba?  Seems a bit odd...
      %{z | mu_ba: @initial_mu * (z.b - z.a)}
    else
      z
    end
  end

  # ---------------------------------------
  # Option handling

  @spec set_tolerance(Keyword.t()) :: Keyword.t()
  defp set_tolerance(opts) do
    # Keyword.put_new_lazy(opts, :tolerance, fn -> Nx.to_number(Nx.Constants.epsilon(opts[:type])) end)
    Keyword.put_new_lazy(opts, :tolerance, fn -> Nx.Constants.epsilon(opts[:type]) |> Nx.to_number() end)
  end

  @spec set_machine_eps(Keyword.t()) :: Keyword.t()
  defp set_machine_eps(opts) do
    # Keyword.put_new_lazy(opts, :machine_eps, fn -> Nx.to_number(Nx.Constants.epsilon(opts[:type])) end)
    Keyword.put_new_lazy(opts, :machine_eps, fn -> Nx.Constants.epsilon(opts[:type]) |> Nx.to_number() end)
  end

  @spec merge_default_opts(Keyword.t()) :: Keyword.t()
  defp merge_default_opts(opts) do
    opts |> set_tolerance() |> set_machine_eps()
  end

  @doc """
  Returns the sign of the tensor as -1 or 1 (or 0 for zero tensors)
  """
  @spec sign(float()) :: float()
  def sign(x) when x < 0.0, do: -1.0
  def sign(x) when x > 0.0, do: 1.0
  def sign(_x), do: 0.0

  @doc """
  Given a list of elements, create a new list with only the unique elements from the list
  """
  @spec unique(list()) :: list()
  def unique(values) do
    MapSet.new(values) |> MapSet.to_list() |> Enum.sort()
  end
end
