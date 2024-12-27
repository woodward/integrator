defmodule Integrator.NonLinearEqnRoot.InternalComputations do
  @moduledoc """
  Functions which are internal or private to `NonLinearEqnRoot`.  These would have been just implemented as private
  functions in the `NonLinearEqnRoot` module, but then they could not be tested, as
  [Patch's feature for testing private functions](https://hexdocs.pm/patch/Patch.html#private/1) does
  not seem to work for `defnp` functions, only `defp` functions.
  """

  import Nx.Defn
  alias Integrator.NonLinearEqnRootRefactor

  alias Integrator.NonLinearEqnRoot.BracketingFailureError
  # alias Integrator.NonLinearEqnRoot.InvalidInitialBracketError
  alias Integrator.NonLinearEqnRoot.MaxFnEvalsExceededError
  alias Integrator.NonLinearEqnRoot.MaxIterationsExceededError

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

  @spec number_of_unique_values(Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  defn number_of_unique_values(one, two, three, four) do
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

  @spec compute_iteration_two_or_three(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defn compute_iteration_two_or_three(z) do
    c =
      case number_of_unique_values(z.fa, z.fb, z.fd, z.fe) do
        4 ->
          interpolate_inverse_cubic(z)

        _length ->
          # Seems like length will always be less than 4 if you're reaching here:
          # if length < 4 or Nx.sign(z.c - z.a) * Nx.sign(z.c - z.b) > 0 do
          if Nx.sign(z.c - z.a) * Nx.sign(z.c - z.b) > 0 do
            interpolate_quadratic_plus_newton(z)
          else
            # what do we do here?  it's not handled in fzero.m...
            z.c
          end
      end

    %{z | iter_type: z.iter_type + 1, c: c}
  end

  @spec bracket(NonLinearEqnRootRefactor.t()) :: {Nx.t(), NonLinearEqnRootRefactor.t()}
  defn bracket(z) do
    continue = 0
    halt = 1

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
            {halt, raise_bracketing_failure_error(z)}
          end
        end
      end

    {status, z}
  end

  @spec raise_bracketing_failure_error(NonLinearEqnRootRefactor.t()) :: NonLinearEqnRootRefactor.t()
  defnp raise_bracketing_failure_error(z) do
    hook(z, fn step -> raise BracketingFailureError, step: step end)
  end

  @spec fn_eval_new_point(NonLinearEqnRootRefactor.t(), NonLinearEqnRootRefactor.zero_fn_t(), Keyword.t()) ::
          NonLinearEqnRootRefactor.t()
  defn fn_eval_new_point(z, zero_fn, opts) do
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
    |> max_iteration_count_exceeded?(opts[:max_iterations])
    |> max_fn_eval_count_exceeded?(opts[:max_fn_eval_count])
  end

  defnp max_iteration_count_exceeded?(z, max_iterations) do
    if z.iteration_count > max_iterations do
      hook(z, fn step -> raise MaxIterationsExceededError, step: step, iteration_count: step.iteration_count end)
    else
      z
    end
  end

  defnp max_fn_eval_count_exceeded?(z, max_fn_eval_count) do
    if z.fn_eval_count > max_fn_eval_count do
      hook(z, fn step -> raise MaxFnEvalsExceededError, step: step, fn_eval_count: step.fn_eval_count end)
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
