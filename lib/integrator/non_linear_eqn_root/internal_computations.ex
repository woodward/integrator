defmodule Integrator.NonLinearEqnRoot.InternalComputations do
  @moduledoc """
  Functions which are internal or private to `NonLinearEqnRoot`.  These would have been just implemented as private
  functions in the `NonLinearEqnRoot` module, but then they could not be tested, as
  [Patch's feature for testing private functions](https://hexdocs.pm/patch/Patch.html#private/1) does
  not seem to work for `defnp` functions, only `defp` functions.
  """

  import Nx.Defn

  alias Integrator.NonLinearEqnRoot.BracketingFailureError
  alias Integrator.NonLinearEqnRoot.IncorrectIterationTypeError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError
  alias Integrator.NonLinearEqnRootRefactor
  alias Integrator.NonLinearEqnRootRefactor.NxOptions

  # This also shows up in Integrator.NonLinearEqnRootRefactor - how can I get rid of the duplication?
  @initial_mu 0.5

  @spec iterate(NonLinearEqnRootRefactor.t(), NonLinearEqnRootRefactor.zero_fn_t(), NxOptions.t()) :: NonLinearEqnRootRefactor.t()
  defn iterate(z, zero_fn, options) do
    continue? = Nx.tensor(1, type: :u8)

    # For debugging:
    # z = print_z(z)

    {z, _, _} =
      while {z, options, continue?}, continue? do
        {status_1, z} =
          z
          |> compute_iteration()
          |> adjust_if_too_close_to_a_or_b(options.machine_eps, options.tolerance)
          |> fn_eval_new_point(zero_fn, options)
          |> check_for_non_monotonicity()
          |> bracket()

        z =
          z
          |> skip_bisection_if_successful_reduction()
          |> update_u()

        # # |> call_output_fn(opts[:nonlinear_eqn_root_output_fn])

        status_2 = converged?(z, options.machine_eps, options.tolerance)
        continue? = not status_2 and status_1

        # For debugging:
        # z = print_z(z)
        {z, options, continue?}
      end

    %{z | x: z.u, fx: z.fu}
  end

  @spec compute_iteration(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn compute_iteration(z) do
    iter_type = z.iter_type

    # How can I get rid of these nasty nested if statements and do a case statement? See attempts below:
    if iter_type == 1 do
      compute_iteration_type_one(z)
    else
      if iter_type == 2 or iter_type == 3 do
        compute_iteration_types_two_or_three(z)
      else
        if iter_type == 4 do
          compute_iteration_type_four(z)
        else
          if iter_type == 5 do
            compute_iteration_type_five(z)
          else
            # Should never reach here:
            hook(z, &raise(IncorrectIterationTypeError, step: &1, iter_type: &1.iter_type))
          end
        end
      end
    end

    # I should be able to do this as a case statement on z.iter_type - why does this not work???

    # First try:
    # case z.iter_type do
    #   1 -> compute_iteration_type_one(z)
    #   2 -> compute_iteration_types_two_or_three(z)
    #   3 -> compute_iteration_types_two_or_three(z)
    #   4 -> compute_iteration_type_four(z)
    #   5 -> compute_iteration_type_five(z)
    # end

    # ---------------

    # 2nd try:

    # one = Nx.tensor(1, type: :s32)
    # two = Nx.tensor(2, type: :s32)
    # three = Nx.tensor(3, type: :s32)
    # four = Nx.tensor(4, type: :s32)
    # five = Nx.tensor(5, type: :s32)

    # case z.iter_type do
    #   ^one -> compute_iteration_type_one(z)
    #   ^two -> compute_iteration_types_two_or_three(z)
    #   ^three -> compute_iteration_types_two_or_three(z)
    #   ^four -> compute_iteration_type_four(z)
    #   ^five -> compute_iteration_type_five(z)
    # end
  end

  @spec compute_iteration_type_one(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn compute_iteration_type_one(z) do
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
      if Nx.abs(z.fa) <= 1000 * Nx.abs(z.fb) and Nx.abs(z.fb) <= 1000 * Nx.abs(z.fa) do
        interpolate_secant(z)
      else
        interpolate_bisect(z)
      end

    %{z | c: c, d: z.u, fd: z.fu, iter_type: 5}
  end

  @spec compute_iteration_types_two_or_three(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn compute_iteration_types_two_or_three(z) do
    c =
      case number_of_unique_values(z.fa, z.fb, z.fd, z.fe) do
        4 ->
          interpolate_inverse_cubic(z)

        _length ->
          # The following line seems wrong: it seems like length will always be less than 4 if you're reaching here:
          # if length < 4 or Nx.sign(z.c - z.a) * Nx.sign(z.c - z.b) > 0 do

          # Shouldn't it be this instead?
          if Nx.sign(z.c - z.a) * Nx.sign(z.c - z.b) > 0 do
            #
            interpolate_quadratic_plus_newton(z)
          else
            # what do we do here?  it's not handled in fzero.m...
            interpolate_quadratic_plus_newton(z)
            # z.c
          end
      end

    %{z | iter_type: z.iter_type + 1, c: c}
  end

  @spec compute_iteration_type_four(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn compute_iteration_type_four(z) do
    # Octave:
    #   # Double secant step.
    #   c = u - 2*(b - a)/(fb - fa)*fu;
    #   # Bisect if too far.
    #   if (abs (c - u) > 0.5*(b - a))
    #     c = 0.5 * (b + a);
    #   endif
    #   iter_type = 5;

    c = interpolate_double_secant(z)

    c =
      if too_far?(c, z) do
        # Bisect if too far:
        interpolate_bisect(z)
      else
        c
      end

    %{z | iter_type: 5, c: c}
  end

  @spec compute_iteration_type_five(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn compute_iteration_type_five(z) do
    # Octave:
    #   # Bisection step.
    #   c = 0.5 * (b + a);
    #   iter_type = 2;

    c = interpolate_bisect(z)
    %{z | iter_type: 2, c: c}
  end

  # For debugging purposes
  @spec print_z(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn print_z(z) do
    hook(z, fn step ->
      print = &inspect(Nx.to_number(&1))

      z_data = """
      %Integrator.NonLinearEqnRootRefactor{
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
          iter_type: #{print.(step.iter_type)}
      }
      """

      IO.puts(z_data)
      step
    end)
  end

  @spec converged?(NonLinearEqnRootRefactor.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn converged?(z, machine_eps, tolerance) do
    z.b - z.a <= 2 * (2 * Nx.abs(z.u) * machine_eps + tolerance)
  end

  @spec too_far?(Nx.t(), NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn too_far?(c, z) do
    Nx.abs(c - z.u) > 0.5 * (z.b - z.a)
  end

  # Modification 2: skip inverse cubic interpolation if nonmonotonicity is detected
  @spec check_for_non_monotonicity(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn check_for_non_monotonicity(z) do
    if Nx.sign(z.fc - z.fa) * Nx.sign(z.fc - z.fb) >= 0 do
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

  @spec find_2nd_starting_point(NonLinearEqnRootRefactor.zero_fn_t(), Nx.t()) :: map()
  defn find_2nd_starting_point(zero_fn, a) do
    # For very small values, switch to absolute rather than relative search:
    a =
      if Nx.abs(a) < 0.001 do
        if a == 0, do: 0.1, else: Nx.sign(a) * 0.1
      else
        a
      end

    fa = zero_fn.(a)
    x = %SearchFor2ndPoint{a: a, fa: fa, b: a, fb: fa, fn_eval_count: 1}
    nx_type = Nx.type(a)
    search_values = Nx.tensor(@search_values, type: nx_type)
    number_of_search_values = Nx.axis_size(search_values, 0)

    # Search in an ever-widening range around the initial point:
    {found_x, _, _} =
      while {x, search_values, i = 0}, not found?(x) and i <= number_of_search_values - 1 do
        search = search_values[i]
        b = x.a + x.a * search
        fb = zero_fn.(b)
        x = %{x | b: b, fb: fb, fn_eval_count: x.fn_eval_count + 1}
        {x, search_values, i + 1}
      end

    found_x
  end

  @spec found?(search_for_2nd_point_t()) :: Nx.t()
  defn found?(x) do
    Nx.sign(x.fa) * Nx.sign(x.fb) <= 0
  end

  @spec skip_bisection_if_successful_reduction(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn skip_bisection_if_successful_reduction(z) do
    # Octave:
    #   if (iter_type == 5 && (b - a) <= mba)
    #     iter_type = 2;
    #   endif
    #   if (iter_type == 2)
    #     mba = mu * (b - a);
    #   endif

    z =
      if z.iter_type == 5 and z.b - z.a <= z.mu_ba do
        %{z | iter_type: 2}
      else
        z
      end

    if z.iter_type == 2 do
      # Should this really be @initial_mu here?  or should it be mu_ba?  Seems a bit odd...
      %{z | mu_ba: (z.b - z.a) * @initial_mu}
    else
      z
    end
  end

  @spec update_u(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
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
    # There's got to be a better, Nx-ey way to do this! This is brute-force (for now).

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

  @spec bracket(NonLinearEqnRootRefactor.t()) :: {Nx.t(), NonLinearEqnRootRefactor.t()}
  defn bracket(z) do
    continue = 1
    halt = 0

    {status, z} =
      if Nx.sign(z.fa) * Nx.sign(z.fc) < 0 do
        # Move c to b:
        {continue, %{z | d: z.b, fd: z.fb, b: z.c, fb: z.fc}}
      else
        if Nx.sign(z.fb) * Nx.sign(z.fc) < 0 do
          {continue, %{z | d: z.a, fd: z.fa, a: z.c, fa: z.fc}}
        else
          if z.fc == 0.0 do
            {halt, %{z | a: z.c, b: z.c, fa: z.fc, fb: z.fc}}
          else
            # Should never reach here
            {halt, hook(z, &raise(BracketingFailureError, step: &1))}
          end
        end
      end

    {status, z}
  end

  @spec fn_eval_new_point(NonLinearEqnRootRefactor.t(), NonLinearEqnRootRefactor.zero_fn_t(), Keyword.t()) ::
          NonLinearEqnRootRefactor.t()
  defn fn_eval_new_point(z, zero_fn, options) do
    fc = zero_fn.(z.c)

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

  @spec raise_if_max_iteration_count_exceeded(NonLinearEqnRootRefactor.t(), Nx.t()) :: NonLinearEqnRootRefactor.t()
  defnp raise_if_max_iteration_count_exceeded(z, max_iterations) do
    if z.iteration_count > max_iterations do
      hook(z, &raise(MaxIterationsExceededError, step: &1, iteration_count: &1.iteration_count))
    else
      z
    end
  end

  @spec raise_if_max_fn_eval_count_exceeded(NonLinearEqnRootRefactor.t(), Nx.t()) :: NonLinearEqnRootRefactor.t()
  defnp raise_if_max_fn_eval_count_exceeded(z, max_fn_eval_count) do
    if z.fn_eval_count > max_fn_eval_count do
      hook(z, &raise(MaxFnEvalsExceededError, step: &1, fn_eval_count: &1.fn_eval_count))
    else
      z
    end
  end

  @spec adjust_if_too_close_to_a_or_b(NonLinearEqnRootRefactor.t(), Nx.t(), Nx.t()) :: NonLinearEqnRootRefactor.t()
  defn adjust_if_too_close_to_a_or_b(z, machine_eps, tolerance) do
    delta = 2 * 0.7 * (2 * Nx.abs(z.u) * machine_eps + tolerance)

    c =
      if z.b - z.a <= 2 * delta do
        (z.a + z.b) / 2
      else
        max(z.a + delta, min(z.b - delta, z.c))
      end

    %{z | c: c}
  end

  @spec interpolate_quadratic_plus_newton(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_quadratic_plus_newton(z) do
    a0 = z.fa
    a1 = (z.fb - z.fa) / (z.b - z.a)
    a2 = ((z.fd - z.fb) / (z.d - z.b) - a1) / (z.d - z.a)

    ## Modification 1: this is simpler and does not seem to be worse.
    c = z.a - a0 / a1

    if a2 != 0 do
      {_z, _a0, _a1, _a2, c, _i} =
        while {z, a0, a1, a2, c, i = 1}, Nx.less_equal(i, z.iter_type) do
          pc = a0 + (a1 + a2 * (c - z.b)) * (c - z.a)
          pdc = a1 + a2 * (2 * c - z.a - z.b)

          new_c =
            if pdc == 0 do
              # Octave does a break here - is the c = 0 caught downstream? Need to handle this case somehow"
              # Note that there is NO test case for this case, as I couldn't figure out how to set up
              # the initial conditions to reach here
              z.a - a0 / a1
            else
              c - pc / pdc
            end

          {z, a0, a1, a2, new_c, i + 1}
        end

      c
    else
      c
    end
  end

  @spec interpolate_inverse_cubic(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_inverse_cubic(z) do
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

  @spec interpolate_double_secant(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_double_secant(z) do
    z.u - 2.0 * (z.b - z.a) / (z.fb - z.fa) * z.fu
  end

  @spec interpolate_bisect(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_bisect(z) do
    0.5 * (z.b + z.a)
  end

  @spec interpolate_secant(NonLinearEqnRootRefactor.t()) :: Nx.t()
  defn interpolate_secant(z) do
    z.u - (z.a - z.b) / (z.fa - z.fb) * z.fu
  end
end
