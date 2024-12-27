defmodule Integrator.NonLinearEqnRootRefactor do
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

  # alias Integrator.NonLinearEqnRoot.InternalComputations, as: Internal
  # alias Integrator.NonLinearEqnRoot.InvalidInitialBracketError
  alias Integrator.NonLinearEqnRoot.TensorTypeError

  @type zero_fn_t :: (Nx.t() -> Nx.t())

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
     :iter_type
   ],
   keep: []}

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
          # Change iter_type to a more descriptive atom later (possibly? or keep it this way??):
          iter_type: Nx.t()
        }

  defstruct a: 0,
            b: 0,
            c: 0,
            d: 0,
            e: 0,
            u: 0,
            #
            # Function evaluations; e.g., fb is fn(b):
            fa: 0,
            fb: 0,
            fc: 0,
            fd: 0,
            fe: 0,
            fu: 0,
            #
            # x (and fx) are the actual found values (i.e., fx should be very close to zero):
            x: 0,
            fx: 0,
            #
            mu_ba: 0,
            #
            fn_eval_count: 0,
            iteration_count: 0,
            # Change iter_type to a more descriptive atom later (possibly?):
            iter_type: 1

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

  @spec find_zero(zero_fn_t(), float() | Nx.t(), float() | Nx.t(), Keyword.t()) :: t()
  deftransform find_zero(zero_fn, a, b, opts \\ []) do
    options = convert_to_nx_options(opts)
    a_nx = a
    b_nx = b
    # a_nx = convert_arg_to_nx_type(a, options.type)
    # b_nx = convert_arg_to_nx_type(b, options.type)
    find_zero_nx(zero_fn, a_nx, b_nx, options)
  end

  defn find_zero_nx(_zero_fn, _a, _b, _options) do
    # fa = zero_fn.(a)
    # fb = zero_fn.(b)
    # fn_eval_count = 2 + fn_evals
    # {u, fu} = if abs(fa) < abs(fb), do: {a, fa}, else: {b, fb}
    # {a, b, fa, fb} = if b < a, do: {b, a, fb, fa}, else: {a, b, fa, fb}

    # z = %__MODULE__{
    #   a: a,
    #   b: b,
    #   d: u,
    #   e: u,
    #   u: u,
    #   #
    #   fa: fa,
    #   fb: fb,
    #   fd: fu,
    #   fe: fu,
    #   fu: fu,
    #   #
    #   fn_eval_count: fn_eval_count,
    #   iter_type: 1,
    #   mu_ba: @initial_mu * (b - a)
    # }

    # if sign(z.fa) * sign(z.fb) > 0.0, do: raise(InvalidInitialBracketError, step: z)

    # case converged?(z, opts[:machine_eps], opts[:tolerance]) do
    #   :continue -> iterate(z, :continue, zero_fn, opts)
    #   :halt -> %{z | x: u, fx: fu}
    # end

    %__MODULE__{}
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

  @spec convert_arg_to_nx_type(Nx.Tensor.t() | float() | integer(), Nx.Type.t()) :: Nx.t()
  def convert_arg_to_nx_type(%Nx.Tensor{} = arg, type) do
    if Nx.type(arg) != type, do: raise(TensorTypeError)
    arg
  end

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
