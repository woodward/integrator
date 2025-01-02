defmodule Integrator.NonLinearEqnRoot do
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

  import Nx.Defn

  alias Integrator.NonLinearEqnRoot.InternalComputations
  alias Integrator.NonLinearEqnRoot.InvalidInitialBracketError
  alias Integrator.NonLinearEqnRoot.TensorTypeError

  @type zero_fn_t :: (Nx.t(), [Nx.t()] -> Nx.t())
  @type output_fn_t :: (Nx.t() -> any())

  @type iteration_type :: 1 | 2 | 3 | 4 | 5
  @initial_mu 0.5

  @derive {Nx.Container,
   containers: [
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
     :x,
     :fx,
     #
     :mu_ba,
     #
     :fn_eval_count,
     :iteration_count,
     :iteration_type,
     :interpolation_type_debug_only
   ],
   keep: [
     :nonlinear_eqn_root_output_fn
   ]}

  @type t :: %__MODULE__{
          a: Nx.t(),
          b: Nx.t(),
          c: Nx.t(),
          d: Nx.t(),
          e: Nx.t(),
          u: Nx.t(),
          #
          # Function evaluations; e.g., fb is fn(b):
          fa: Nx.t(),
          fb: Nx.t(),
          fc: Nx.t(),
          fd: Nx.t(),
          fe: Nx.t(),
          fu: Nx.t(),
          #
          # x (and fx) are the actual found values (i.e., fx should be very close to zero):
          x: Nx.t(),
          fx: Nx.t(),
          #
          mu_ba: Nx.t(),
          #
          fn_eval_count: Nx.t(),
          iteration_count: Nx.t(),
          # Change iteration_type to a more descriptive atom later (possibly? or keep it this way??):
          iteration_type: Nx.t(),
          interpolation_type_debug_only: Nx.t(),
          #
          nonlinear_eqn_root_output_fn: output_fn_t()
        }

  defstruct a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            u: 0.0,
            #
            # Function evaluations; e.g., fb is fn(b):
            fa: 0.0,
            fb: 0.0,
            fc: 0.0,
            fd: 0.0,
            fe: 0.0,
            fu: 0.0,
            #
            # x (and fx) are the actual found values (i.e., fx should be very close to zero):
            x: 0.0,
            fx: 0.0,
            #
            mu_ba: 0.0,
            #
            fn_eval_count: 0,
            iteration_count: 0,
            # Change iteration_type to a more descriptive atom later (possibly?):
            iteration_type: 1,
            interpolation_type_debug_only: 0,
            #
            nonlinear_eqn_root_output_fn: nil

  defmodule NxOptions do
    @moduledoc """
    `NimbleOptions` converted into an Nx-friendly format for use when finding the non-linear eqn root
    """

    @derive {Nx.Container,
             containers: [
               :max_iterations,
               :max_fn_eval_count,
               :machine_eps,
               :tolerance
             ],
             keep: [
               :type,
               :nonlinear_eqn_root_output_fn
             ]}

    @type t :: %__MODULE__{
            max_iterations: Nx.t(),
            max_fn_eval_count: Nx.t(),
            type: Nx.Type.t(),
            machine_eps: Nx.t(),
            tolerance: Nx.t(),
            nonlinear_eqn_root_output_fn: fun()
          }

    defstruct max_iterations: 1000,
              max_fn_eval_count: 1000,
              type: {:f, 64},
              machine_eps: 0,
              tolerance: 0,
              nonlinear_eqn_root_output_fn: nil
  end

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
  def options_schema, do: @options_schema

  @doc """
  Finds a zero for a function in an interval `[a, b]` (if the 2nd argument is a list) or
  in the vicinity of `a` (if the 2nd argument is a float).

  ## Options

  #{NimbleOptions.docs(@options_schema)}

  """

  @spec find_zero(zero_fn_t(), float() | Nx.t(), float() | Nx.t(), [float() | Nx.t()], Keyword.t()) :: t()
  deftransform find_zero(zero_fn, a, b, zero_fn_args, opts \\ []) do
    options = convert_to_nx_options(opts)
    a_nx = convert_arg_to_nx_type(a, options.type)
    b_nx = convert_arg_to_nx_type(b, options.type)
    zero_fn_args_nx = zero_fn_args |> Enum.map(&convert_arg_to_nx_type(&1, options.type))

    find_zero_nx(zero_fn, a_nx, b_nx, zero_fn_args_nx, options)
  end

  @spec find_zero_with_single_point(zero_fn_t(), float() | Nx.t(), [float() | Nx.t()], Keyword.t()) :: t()
  deftransform find_zero_with_single_point(zero_fn, solo_point, zero_fn_args, opts \\ []) do
    options = convert_to_nx_options(opts)
    solo_point_nx = convert_arg_to_nx_type(solo_point, options.type)
    second_point = InternalComputations.find_2nd_starting_point(zero_fn, solo_point_nx, zero_fn_args)
    zero_fn_args_nx = zero_fn_args |> Enum.map(&convert_arg_to_nx_type(&1, options.type))

    result = find_zero(zero_fn, solo_point_nx, second_point.b, zero_fn_args_nx, opts)
    %{result | fn_eval_count: Nx.add(result.fn_eval_count, second_point.fn_eval_count)}
  end

  @spec find_zero_nx(zero_fn_t(), Nx.t(), Nx.t(), [Nx.t()], NxOptions.t()) :: t()
  defn find_zero_nx(zero_fn, a, b, zero_fn_args, options) do
    fa = zero_fn.(a, zero_fn_args)
    fb = zero_fn.(b, zero_fn_args)
    # fn_eval_count = 2 + fn_evals
    fn_eval_count = 2
    {u, fu} = if Nx.abs(fa) < Nx.abs(fb), do: {a, fa}, else: {b, fb}
    {a, b, fa, fb} = if b < a, do: {b, a, fb, fa}, else: {a, b, fa, fb}
    c = Nx.tensor(0.0, type: options.type)
    fc = Nx.tensor(0.0, type: options.type)
    x = Nx.tensor(0.0, type: options.type)
    fx = Nx.tensor(0.0, type: options.type)

    # These don't seem to work:
    # c = Nx.Constants.nan(type: options.type)
    # fc = Nx.Constants.nan(type: options.type)
    # x = Nx.Constants.nan(type: options.type)
    # fx = Nx.Constants.nan(type: options.type)

    z = %__MODULE__{
      a: a,
      b: b,
      c: c,
      d: u,
      e: u,
      u: u,
      x: x,
      #
      fa: fa,
      fb: fb,
      fc: fc,
      fd: fu,
      fe: fu,
      fu: fu,
      fx: fx,
      #
      fn_eval_count: fn_eval_count,
      iteration_type: 1,
      mu_ba: (b - a) * @initial_mu,
      nonlinear_eqn_root_output_fn: options.nonlinear_eqn_root_output_fn
    }

    z = if Nx.sign(z.fa) * Nx.sign(z.fb) > 0.0, do: hook(z, &raise(InvalidInitialBracketError, step: &1)), else: z

    # case converged?(z, opts[:machine_eps], opts[:tolerance]) do
    #   :continue -> iterate(z, :continue, zero_fn, opts)
    #   :halt -> %{z | x: u, fx: fu}
    # end

    InternalComputations.iterate(z, zero_fn, zero_fn_args, options)
    # z
  end

  # def find_zero(zero_fn, solo_point, opts, _fn_evals) do
  #   # second_point = Internal.find_2nd_starting_point(zero_fn, solo_point)

  #   # find_zero(zero_fn, [solo_point, second_point.b], opts, second_point.fn_eval_count)
  # end

  @spec bracket_x(t()) :: {Nx.t(), Nx.t()}
  def bracket_x(z) do
    {z.a, z.b}
  end

  @spec bracket_fx(t()) :: {Nx.t(), Nx.t()}
  def bracket_fx(z) do
    {z.fa, z.fb}
  end

  def option_keys, do: NimbleOptions.validate!([], @options_schema) |> Keyword.keys()

  # Convert the following to private functions and test with Patch:

  @spec convert_arg_to_nx_type(Nx.Tensor.t() | float() | integer() | fun(), Nx.Type.t()) :: Nx.t()
  def convert_arg_to_nx_type(%Nx.Tensor{} = arg, type) do
    if Nx.type(arg) != type, do: raise(TensorTypeError)
    arg
  end

  def convert_arg_to_nx_type(arg, _type) when is_function(arg), do: arg
  def convert_arg_to_nx_type(arg, type), do: Nx.tensor(arg, type: type)

  @spec convert_to_nx_options(Keyword.t()) :: NxOptions.t()
  def convert_to_nx_options(opts) do
    nimble_opts = opts |> NimbleOptions.validate!(@options_schema) |> Map.new()
    nx_type = nimble_opts[:type] |> Nx.Type.normalize!()

    machine_eps =
      if Map.has_key?(nimble_opts, :machine_eps) do
        Nx.tensor(Map.get(nimble_opts, :machine_eps), type: nx_type)
      else
        Nx.Constants.epsilon(nx_type)
      end

    tolerance =
      if Map.has_key?(nimble_opts, :tolerance) do
        Nx.tensor(Map.get(nimble_opts, :tolerance), type: nx_type)
      else
        Nx.Constants.epsilon(nx_type)
      end

    %NxOptions{
      machine_eps: machine_eps,
      max_fn_eval_count: nimble_opts[:max_fn_eval_count],
      max_iterations: nimble_opts[:max_iterations],
      nonlinear_eqn_root_output_fn: nimble_opts[:nonlinear_eqn_root_output_fn],
      tolerance: tolerance,
      type: nx_type
    }
  end
end
