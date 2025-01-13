defmodule Integrator.NonLinearEqnRoot.InternalComputations do
  @moduledoc """
  Functions which are internal or private to `NonLinearEqnRoot`.  These would have been just implemented as private
  functions in the `NonLinearEqnRoot` module, but then they could not be tested.
  """

  import Nx.Defn

  import Integrator.Utils,
    only: [
      different_signs?: 2,
      different_signs_or_any_zeros?: 2,
      same_signs?: 2,
      same_signs_or_any_zeros?: 2
    ]

  alias Integrator.ExternalFnAdapter
  alias Integrator.Interpolation
  alias Integrator.NonLinearEqnRoot
  alias Integrator.NonLinearEqnRoot.BracketingFailureError
  alias Integrator.NonLinearEqnRoot.IncorrectIterationTypeError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError
  alias Integrator.NonLinearEqnRoot.NxOptions

  @initial_mu 0.5
  defn initial_mu, do: @initial_mu

  # These values are here just to aid in debugging (they show up when printing out the main struct)
  @interpolation_bisect 0
  @interpolation_double_secant 1
  @interpolation_double_secant_plus_bisect 2
  @interpolation_inverse_cubic 3
  @interpolation_none 4
  @interpolation_quadratic_plus_newton 5
  @interpolation_secant 6
  @interpolation_undetermined 7

  @interpolation_types_for_debug_only %{
    @interpolation_bisect => :bisect,
    @interpolation_double_secant => :double_secant,
    @interpolation_double_secant_plus_bisect => :double_secant_plus_bisect,
    @interpolation_inverse_cubic => :inverse_cubic,
    @interpolation_none => :none,
    @interpolation_quadratic_plus_newton => :quadratic_plus_newton,
    @interpolation_secant => :secant,
    @interpolation_undetermined => :undetermined
  }

  # Continue searching for a root:
  @continue 1

  # Halt - a root has been found:
  @halt 0

  @spec iterate(NonLinearEqnRoot.t(), NonLinearEqnRoot.zero_fn_t(), [Nx.t()], NxOptions.t()) ::
          NonLinearEqnRoot.t()
  defn iterate(z, zero_fn, zero_fn_args, options) do
    continue? = Nx.tensor(1, type: :u8)

    # For debugging:
    # z = print_z(z)

    zero_fn_args_as_tuple = zero_fn_args |> to_tuple()

    {z, _, _, _} =
      while {z, options, continue?, zero_fn_args_as_tuple}, continue? do
        {status_1, z} =
          z
          |> compute_iteration()
          |> adjust_if_too_close_to_a_or_b(options.machine_eps, options.tolerance)
          |> fn_eval_new_point(zero_fn, zero_fn_args_as_tuple, options)
          |> check_for_non_monotonicity()
          |> bracket()

        z =
          z
          |> skip_bisection_if_successful_reduction()
          |> update_u()
          |> then(&ExternalFnAdapter.invoke_external_fn(&1, options.output_fn_adapter))

        status_2 = converged?(z, options.machine_eps, options.tolerance)
        continue? = not status_2 and status_1

        # For debugging:
        # z = print_z(z)
        {z, options, continue?, zero_fn_args_as_tuple}
      end

    # I'm not 100% sure why this is not needed - is it happening somewhere else after the big refactor?
    # %{z | x: z.u, fx: z.fu}
    z
  end

  deftransform to_tuple(list_of_args) do
    list_of_args |> List.to_tuple()
  end

  @spec compute_iteration(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn compute_iteration(z) do
    iteration_type = z.iteration_type

    # How can I get rid of these nasty nested if statements and do a case statement instead? See failed attempts below:
    if iteration_type == 1 do
      compute_iteration_type_1(z)
    else
      if iteration_type == 2 or iteration_type == 3 do
        compute_iteration_types_2_or_3(z)
      else
        if iteration_type == 4 do
          compute_iteration_type_4(z)
        else
          if iteration_type == 5 do
            compute_iteration_type_5(z)
          else
            # Should never reach here:
            hook(z, &raise(IncorrectIterationTypeError, step: &1, iteration_type: &1.iteration_type))
          end
        end
      end
    end

    # ----------------------------------------------------------------------------------------------
    # Failed attempts to do a case statement:

    # I should be able to do this as a case statement on z.iteration_type - why does this not work???

    # First try:
    # case z.iteration_type do
    #   1 -> compute_iteration_type_1(z)
    #   2 -> compute_iteration_types_2_or_3(z)
    #   3 -> compute_iteration_types_2_or_3(z)
    #   4 -> compute_iteration_type_4(z)
    #   5 -> compute_iteration_type_5(z)
    # end

    # ---------------

    # 2nd try:

    # one = Nx.tensor(1, type: :s32)
    # two = Nx.tensor(2, type: :s32)
    # three = Nx.tensor(3, type: :s32)
    # four = Nx.tensor(4, type: :s32)
    # five = Nx.tensor(5, type: :s32)

    # case z.iteration_type do
    #   ^one -> compute_iteration_type_1(z)
    #   ^two -> compute_iteration_types_2_or_3(z)
    #   ^three -> compute_iteration_types_2_or_3(z)
    #   ^four -> compute_iteration_type_4(z)
    #   ^five -> compute_iteration_type_5(z)
    # end
  end

  @spec compute_iteration_type_1(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn compute_iteration_type_1(z) do
    # Octave:
    #   if (abs (fa) <= 1e3*abs (fb) && abs (fb) <= 1e3*abs (fa))
    #     # Secant step.
    #     c = u - (a - b) / (fa - fb) * fu;
    #   else
    #     # Bisection step.
    #     c = 0.5*(a + b);
    #   endif
    #   d = u; fd = fu;
    #   iteration_type = 5;

    # What is the significance or meaning of the 1000 here? Replace with a more descriptive module variable
    {c, interpolation_type} =
      if Nx.abs(z.fa) <= 1000 * Nx.abs(z.fb) and Nx.abs(z.fb) <= 1000 * Nx.abs(z.fa) do
        {Interpolation.secant(z.a, z.fa, z.b, z.fb, z.u, z.fu), @interpolation_secant}
      else
        {Interpolation.bisect(z.a, z.b), @interpolation_bisect}
      end

    %{z | c: c, d: z.u, fd: z.fu, iteration_type: 5, interpolation_type_debug_only: interpolation_type}
  end

  @spec compute_iteration_types_2_or_3(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn compute_iteration_types_2_or_3(z) do
    length = number_of_unique_values(z.fa, z.fb, z.fd, z.fe)

    {c, interpolation_type} =
      if length == 4 do
        {Interpolation.inverse_cubic(z.a, z.fa, z.b, z.fb, z.d, z.fd, z.e, z.fe), @interpolation_inverse_cubic}
      else
        # The following line seems wrong: it seems like length will always be less than 4 if you're reaching here:
        # if length < 4 or same_signs?(z.c - z.a, z.c - z.b) do
        #
        # Shouldn't it be this instead?
        if same_signs?(z.c - z.a, z.c - z.b) do
          #
          {Interpolation.quadratic_plus_newton(z.a, z.fa, z.b, z.fb, z.d, z.fd, z.iteration_type),
           @interpolation_quadratic_plus_newton}
        else
          # what do we do here?  it's not handled in fzero.m...
          {Interpolation.quadratic_plus_newton(z.a, z.fa, z.b, z.fb, z.d, z.fd, z.iteration_type),
           @interpolation_quadratic_plus_newton}

          # {z.c, @interpolation_none}
        end
      end

    %{z | iteration_type: z.iteration_type + 1, c: c, interpolation_type_debug_only: interpolation_type}
  end

  @spec compute_iteration_type_4(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn compute_iteration_type_4(z) do
    # Octave:
    #   # Double secant step.
    #   c = u - 2*(b - a)/(fb - fa)*fu;
    #   # Bisect if too far.
    #   if (abs (c - u) > 0.5*(b - a))
    #     c = 0.5 * (b + a);
    #   endif
    #   iteration_type = 5;

    c = Interpolation.double_secant(z.a, z.fa, z.b, z.fb, z.u, z.fu)

    {c, interpolation_type} =
      if too_far?(c, z) do
        # Bisect if too far:
        {Interpolation.bisect(z.a, z.b), @interpolation_double_secant_plus_bisect}
      else
        {c, @interpolation_double_secant}
      end

    %{z | iteration_type: 5, c: c, interpolation_type_debug_only: interpolation_type}
  end

  @spec compute_iteration_type_5(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn compute_iteration_type_5(z) do
    # Octave:
    #   # Bisection step.
    #   c = 0.5 * (b + a);
    #   iteration_type = 2;

    c = Interpolation.bisect(z.a, z.b)
    %{z | iteration_type: 2, c: c, interpolation_type_debug_only: @interpolation_bisect}
  end

  @spec converged?(NonLinearEqnRoot.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn converged?(z, machine_eps, tolerance) do
    z.b - z.a <= 2 * (2 * Nx.abs(z.u) * machine_eps + tolerance)
  end

  @spec too_far?(Nx.t(), NonLinearEqnRoot.t()) :: Nx.t()
  defn too_far?(c, z) do
    Nx.abs(c - z.u) > 0.5 * (z.b - z.a)
  end

  # Modification 2: skip inverse cubic interpolation if nonmonotonicity is detected
  @spec check_for_non_monotonicity(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn check_for_non_monotonicity(z) do
    if same_signs_or_any_zeros?(z.fc - z.fa, z.fc - z.fb) do
      # The new point broke monotonicity.
      # Disable inverse cubic:
      %{z | fe: z.fc}
    else
      %{z | e: z.d, fe: z.fd}
    end
  end

  @search_values [-0.01, 0.025, -0.05, 0.10, -0.25, 0.50, -1.0, 2.5, -5.0, 10.0, -50.0, 100.0, 500.0, 1000.0]

  defmodule SearchFor2ndPoint do
    @moduledoc false
    @derive {Nx.Container,
     containers: [
       :a,
       :b,
       #
       :fa,
       :fb,
       #
       :fn_eval_count
     ],
     keep: []}

    defstruct a: 0,
              b: 0,
              #
              fa: 0,
              fb: 0,
              #
              fn_eval_count: 0
  end

  @type search_for_2nd_point_t :: %SearchFor2ndPoint{
          a: Nx.t(),
          b: Nx.t(),
          #
          # Function evaluations; e.g., fb is fn(b):
          fa: Nx.t(),
          fb: Nx.t(),
          #
          fn_eval_count: Nx.t()
        }

  @spec find_2nd_starting_point(NonLinearEqnRoot.zero_fn_t(), Nx.t(), [Nx.t()]) :: map()
  defn find_2nd_starting_point(zero_fn, a, zero_fn_args) do
    # For very small values, switch to absolute rather than relative search:
    a =
      if Nx.abs(a) < 0.001 do
        if a == 0, do: 0.1, else: Nx.sign(a) * 0.1
      else
        a
      end

    fa = zero_fn.(a, zero_fn_args)
    x = %SearchFor2ndPoint{a: a, fa: fa, b: a, fb: fa, fn_eval_count: 1}
    nx_type = Nx.type(a)
    search_values = Nx.tensor(@search_values, type: nx_type)
    number_of_search_values = Nx.axis_size(search_values, 0)

    # Search in an ever-widening range around the initial point:
    {found_x, _, _} =
      while {x, search_values, i = 0}, not found?(x) and i <= number_of_search_values - 1 do
        search = search_values[i]
        b = x.a + x.a * search
        fb = zero_fn.(b, zero_fn_args)
        x = %{x | b: b, fb: fb, fn_eval_count: x.fn_eval_count + 1}
        {x, search_values, i + 1}
      end

    found_x
  end

  @spec found?(search_for_2nd_point_t()) :: Nx.t()
  defn found?(x), do: different_signs_or_any_zeros?(x.fa, x.fb)

  @spec skip_bisection_if_successful_reduction(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn skip_bisection_if_successful_reduction(z) do
    # Octave:
    #   if (iteration_type == 5 && (b - a) <= mba)
    #     iteration_type = 2;
    #   endif
    #   if (iteration_type == 2)
    #     mba = mu * (b - a);
    #   endif

    z =
      if z.iteration_type == 5 and z.b - z.a <= z.mu_ba do
        %{z | iteration_type: 2}
      else
        z
      end

    if z.iteration_type == 2 do
      # Should this really be @initial_mu here?  or should it be mu_ba?  Seems a bit odd to me...
      %{z | mu_ba: (z.b - z.a) * @initial_mu}
    else
      z
    end
  end

  @spec update_u(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn update_u(z) do
    # Octave:
    #   if (abs (fa) < abs (fb))
    #     u = a; fu = fa;
    #   else
    #     u = b; fu = fb;
    #   endif

    if Nx.abs(z.fa) < Nx.abs(z.fb) do
      %{z | u: z.a, fu: z.fa}
    else
      %{z | u: z.b, fu: z.fb}
    end
  end

  @spec number_of_unique_values(Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn number_of_unique_values(one, two, three, four) do
    # There's got to be a better, Nx-ey way to do this! This is brute-force (for now) - ugh!

    if one == two == three == four do
      1
    else
      one_ne_two = one != two
      one_ne_three = one != three
      one_ne_four = one != four

      two_ne_three = two != three
      two_ne_four = two != four

      three_ne_four = three != four

      one_ne_two + one_ne_three + one_ne_four + two_ne_three + two_ne_four + three_ne_four - 2
    end
  end

  @spec bracket(NonLinearEqnRoot.t()) :: {Nx.t(), NonLinearEqnRoot.t()}
  defn bracket(z) do
    {status, z} =
      if different_signs?(z.fa, z.fc) do
        # Move c to b:
        {@continue, %{z | d: z.b, fd: z.fb, b: z.c, fb: z.fc}}
      else
        if different_signs?(z.fb, z.fc) do
          {@continue, %{z | d: z.a, fd: z.fa, a: z.c, fa: z.fc}}
        else
          if z.fc == 0.0 do
            {@halt, %{z | a: z.c, b: z.c, fa: z.fc, fb: z.fc}}
          else
            # Should never reach here
            {@halt, hook(z, &raise(BracketingFailureError, step: &1))}
          end
        end
      end

    {status, z}
  end

  @spec is_nil?(any()) :: Nx.t()
  deftransform is_nil?(item) do
    if is_nil(item), do: 1, else: 0
  end

  deftransform to_list(tuple_args) do
    tuple_args |> Tuple.to_list()
  end

  @spec fn_eval_new_point(NonLinearEqnRoot.t(), NonLinearEqnRoot.zero_fn_t(), [Nx.t()], Keyword.t()) ::
          NonLinearEqnRoot.t()
  defn fn_eval_new_point(z, zero_fn, zero_fn_args, options) do
    zero_fn_args_as_list = zero_fn_args |> to_list()
    fc = zero_fn.(z.c, zero_fn_args_as_list)

    %{
      z
      | fc: fc,
        x: z.c,
        fx: fc,
        fn_eval_count: z.fn_eval_count + 1,
        # Perhaps move the incrementing of the iteration count elsewhere?
        iteration_count: z.iteration_count + 1
    }
    |> raise_if_max_iteration_count_exceeded(options.max_iterations)
    |> raise_if_max_fn_eval_count_exceeded(options.max_fn_eval_count)
  end

  @spec raise_if_max_iteration_count_exceeded(NonLinearEqnRoot.t(), Nx.t()) :: NonLinearEqnRoot.t()
  defnp raise_if_max_iteration_count_exceeded(z, max_iterations) do
    if z.iteration_count > max_iterations do
      hook(z, &raise(MaxIterationsExceededError, step: &1, iteration_count: &1.iteration_count))
    else
      z
    end
  end

  @spec raise_if_max_fn_eval_count_exceeded(NonLinearEqnRoot.t(), Nx.t()) :: NonLinearEqnRoot.t()
  defnp raise_if_max_fn_eval_count_exceeded(z, max_fn_eval_count) do
    if z.fn_eval_count > max_fn_eval_count do
      hook(z, &raise(MaxFnEvalsExceededError, step: &1, fn_eval_count: &1.fn_eval_count))
    else
      z
    end
  end

  @spec adjust_if_too_close_to_a_or_b(NonLinearEqnRoot.t(), Nx.t(), Nx.t()) :: NonLinearEqnRoot.t()
  defn adjust_if_too_close_to_a_or_b(z, machine_eps, tolerance) do
    type = Nx.type(z.c)
    two = Nx.tensor(2, type: type)
    delta = two * Nx.tensor(0.7, type: type) * (two * Nx.abs(z.u) * machine_eps + tolerance)

    c =
      if z.b - z.a <= two * delta do
        (z.a + z.b) / two
      else
        max(z.a + delta, min(z.b - delta, z.c))
      end

    %{z | c: c}
  end

  # ================================================================================================
  # Some debugging utils below here:

  # For debugging purposes
  @spec print_line(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn print_line(z) do
    hook(z, fn step ->
      IO.puts("# ----------------------------------")
      step
    end)
  end

  # For debugging purposes
  @spec print_number_of_unique_values(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn print_number_of_unique_values(z) do
    hook(z, fn step ->
      IO.puts("# Number of unique values: #{inspect(Nx.to_number(number_of_unique_values(step.fa, step.fb, step.fd, step.fe)))}")
      step
    end)
  end

  # For debugging purposes
  @spec print_computing_iteration_type(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn print_computing_iteration_type(z) do
    # hook(z, fn step ->
    #   IO.puts("Computing iteration type #{inspect(Nx.to_number(step.iteration_type))}")
    #   step
    # end)
    z
  end

  # For debugging purposes
  @spec print_z(NonLinearEqnRoot.t()) :: NonLinearEqnRoot.t()
  defn print_z(z) do
    hook(z, fn step ->
      print = &inspect(Nx.to_number(&1))
      interpolation_type = Map.get(@interpolation_types_for_debug_only, Nx.to_number(step.interpolation_type_debug_only))

      z_data = """
      %Integrator.NonLinearEqnRoot{
          a: #{print.(step.a)},
          b: #{print.(step.b)},
          c: #{print.(step.c)},
          d: #{print.(step.d)},
          e: #{print.(step.e)},
          u: #{print.(step.u)},
          fa: #{print.(step.fa)},
          fb: #{print.(step.fb)},
          fc: #{print.(step.fc)},
          fd: #{print.(step.fd)},
          fe: #{print.(step.fe)},
          fu: #{print.(step.fu)},
          x: #{print.(step.x)},
          fx: #{print.(step.fx)},
          mu_ba: #{print.(step.mu_ba)},
          fn_eval_count: #{print.(step.fn_eval_count)},
          iteration_count: #{print.(step.iteration_count)},
          iteration_type: #{print.(step.iteration_type)},
          interpolation_type_debug_only: :#{interpolation_type}
      }
      """

      IO.puts(z_data)
      step
    end)
  end

  # Used in tests
  @spec bracket_x(NonLinearEqnRoot.t()) :: {Nx.t(), Nx.t()}
  def bracket_x(z) do
    {z.a, z.b}
  end

  # Used in tests
  @spec bracket_fx(NonLinearEqnRoot.t()) :: {Nx.t(), Nx.t()}
  def bracket_fx(z) do
    {z.fa, z.fb}
  end
end
