defmodule Integrator.AdaptiveStepsizeRefactor do
  @moduledoc """
  Integrates a set of ODEs with an adaptive timestep.
  """

  import Nx.Defn

  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.AdaptiveStepsize.InternalComputations
  alias Integrator.ExternalFnAdapter
  alias Integrator.NonLinearEqnRoot
  alias Integrator.Point
  alias Integrator.RungeKutta
  alias Integrator.Utils

  # import Integrator.Utils, only: [convert_arg_to_nx_type: 2, timestamp_μs: 0, elapsed_time_μs: 1, same_signs?: 2]
  import Integrator.Utils, only: [timestamp_μs: 0]

  defmodule NxOptions do
    @moduledoc """
    `NimbleOptions` converted into an Nx-friendly `Nx.Container` struct for use when finding the non-linear eqn root
    (so that the options can be safely passed from Elixir-land to Nx-land).
    """

    @derive {Nx.Container,
     containers: [
       :abs_tol,
       :rel_tol,
       :norm_control?,
       :fixed_output_times?,
       :fixed_output_step,
       :speed,
       :max_step,
       :max_number_of_errors,
       :nx_while_loop_integration?,
       #
       :event_fn_adapter,
       :output_fn_adapter,
       :zero_fn_adapter,
       #
       :non_linear_eqn_root_nx_options
     ],
     keep: [
       :order,
       :refine,
       :type
     ]}

    @type t :: %__MODULE__{
            abs_tol: Nx.t(),
            rel_tol: Nx.Type.t(),
            norm_control?: Nx.t(),
            order: integer(),
            fixed_output_times?: Nx.t(),
            fixed_output_step: Nx.t(),
            speed: Nx.t(),
            refine: integer(),
            type: Nx.Type.t(),
            max_step: Nx.t(),
            max_number_of_errors: Nx.t(),
            nx_while_loop_integration?: Nx.t(),
            #
            event_fn_adapter: ExternalFnAdapter.t(),
            output_fn_adapter: ExternalFnAdapter.t(),
            zero_fn_adapter: ExternalFnAdapter.t(),
            #
            non_linear_eqn_root_nx_options: NonLinearEqnRoot.NxOptions.t()
          }

    # The default values here are just placeholders; the actual defaults come from NimbleOpts
    # (and are then converted to Nx in convert_to_nx_options/3)
    defstruct abs_tol: 1000.0,
              rel_tol: 1000.0,
              norm_control?: Nx.u8(1),
              order: 0,
              fixed_output_times?: Nx.u8(0),
              fixed_output_step: 1000.0,
              speed: Nx.Constants.nan(:f64),
              refine: 0,
              type: {:f, 64},
              max_step: 0.0,
              max_number_of_errors: 0,
              nx_while_loop_integration?: 1,
              #
              event_fn_adapter: %ExternalFnAdapter{},
              output_fn_adapter: %ExternalFnAdapter{},
              zero_fn_adapter: %ExternalFnAdapter{},
              #
              non_linear_eqn_root_nx_options: %NonLinearEqnRoot.NxOptions{}
  end

  # How do I make the NimbleOptions handle both float values and tensors? Right now I'm handling this by
  # setting the :type below to :any, which is not great...
  options = [
    abs_tol: [
      # type: :float,
      type: :any,
      doc: """
      The absolute tolerance used when computing the absolute relative norm. Defaults to 1.0e-06 in the Nx type that's been specified.
      """,
      default: 1.0e-06
    ],
    rel_tol: [
      # type: :float,
      type: :any,
      doc: """
       The relative tolerance used when computing the absolute relative norm. Defaults to 1.0e-03 in the Nx type that's been specified.
      """,
      default: 1.0e-03
    ],
    norm_control?: [
      # type: :boolean,
      type: :any,
      doc: "Indicates whether norm control is to be used when computing the absolute relative norm.",
      default: true
    ],
    fixed_output_times?: [
      # type: :boolean,
      type: :any,
      doc: "Indicates whether output is to be generated at some fixed interval",
      default: false
    ],
    fixed_output_step: [
      # type: :float,
      type: :any,
      doc: "The fixed output timestep",
      default: 0.0
    ],
    type: [
      type: {:in, [:f32, :f64]},
      doc: "The Nx type.",
      default: :f32
    ],
    max_number_of_errors: [
      type: :integer,
      doc: "The maximum number of permissible errors before the integration is halted.",
      default: 5_000
    ],
    max_step: [
      type: :any,
      doc: """
      The default max time step.  The default value is determined by the start and end times.
      """
    ],
    nx_while_loop_integration?: [
      type: :boolean,
      doc: """
      Indicates whether an Nx `while` loop is to be used for the main integration loop or Elixir recursion.
      Note that specifying a playback speed will override this value (setting it to false).
      """,
      default: true
    ],
    refine: [
      type: :pos_integer,
      doc: """
      Indicates the number of additional interpolated points. `1` means no interpolation; `2` means one
      additional interpolated point; etc. Note that this is ignored if there is a fixed time step.
      """,
      default: 4
    ],
    speed: [
      # type: {:or, [:atom, :float]},
      type: :any,
      doc: """
      `:infinite` means to simulate as fast as possible. `1.0` means real time, `2.0` means twice as fast as real time,
      `0.5` means half as fast as real time, etc.
      """,
      default: :infinite
    ],
    event_fn: [
      type: {:or, [{:fun, 2}, nil]},
      doc: "A 2 arity function which determines whether an event has occured.  If so, the integration is halted.",
      default: nil
    ],
    output_fn: [
      type: {:or, [{:fun, 1}, nil]},
      doc: "A 1 arity function which is called at each output point.",
      default: nil
    ],
    zero_fn: [
      type: {:or, [{:fun, 2}, nil]},
      doc: "Finds the zero; used in conjunction with `event_fn`",
      default: nil
    ]
  ]

  @options_schema_adaptive_stepsize_only NimbleOptions.new!(options)
  def options_schema_adaptive_stepsize_only, do: @options_schema_adaptive_stepsize_only

  @options_schema NimbleOptions.new!(NonLinearEqnRoot.options_schema().schema |> Keyword.merge(options))
  def options_schema, do: @options_schema

  @type options_t() :: unquote(NimbleOptions.option_typespec(@options_schema))

  @doc """
  Integrates a set of ODEs.

  ## Options

  #{NimbleOptions.docs(@options_schema_adaptive_stepsize_only)}

  ### Additional Options

  Also see the options for the `Integrator.NonLinearEqnRoot.find_zero/4` which are passed
  into `integrate/10`.

  Originally adapted from the Octave
  [integrate_adaptive.m](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/integrate_adaptive.m)

  See [Wikipedia](https://en.wikipedia.org/wiki/Adaptive_stepsize)
  """
  @spec integrate(
          stepper_fn :: RungeKutta.stepper_fn_t(),
          interpolate_fn :: RungeKutta.interpolate_fn_t(),
          ode_fn :: RungeKutta.ode_fn_t(),
          t_start :: Nx.t(),
          t_end :: Nx.t(),
          initial_tstep :: Nx.t(),
          x0 :: Nx.t(),
          order :: integer(),
          opts :: Keyword.t()
        ) :: IntegrationStep.t()

  deftransform integrate(stepper_fn, interpolate_fn, ode_fn, t_start, t_end, initial_tstep, x0, order, opts \\ []) do
    start_timestamp_μs = timestamp_μs()
    options = convert_to_nx_options(t_start, t_end, order, opts)
    t_end = IntegrationStep.to_tensor(t_end, options.type)

    initial_tstep =
      if initial_tstep do
        initial_tstep
      else
        starting_stepsize(order, ode_fn, t_start, x0, options.abs_tol, options.rel_tol, options.norm_control?)
      end

    initial_step =
      IntegrationStep.new(stepper_fn, interpolate_fn, ode_fn, t_start, initial_tstep, x0, options, start_timestamp_μs)

    # Broadcast the initial conditions (t_start & x0) as the first output point (if there is an output function):
    %Point{t: initial_step.t_current, x: initial_step.x_current} |> options.output_fn_adapter.external_fn.()

    if options.nx_while_loop_integration? == Nx.u8(1) do
      InternalComputations.integrate_via_nx_while_loop(initial_step, t_end, options)
    else
      InternalComputations.integrate_via_elixir_recursion(initial_step, t_end, options)
    end
    |> InternalComputations.record_elapsed_time()
  end

  @doc """
  Computes a good initial timestep for an ODE solver of order `order`
  using the algorithm described in the reference below.

  The input argument `ode_fn`, is the function describing the differential
  equations, `t0` is the initial time, and `x0` is the initial
  condition.  `abs_tol` and `rel_tol` are the absolute and relative
  tolerance on the ODE integration.

  Originally based on the Octave
  [`starting_stepsize.m`](https://github.com/gnu-octave/octave/blob/default/scripts/ode/private/starting_stepsize.m).

  Reference:

  E. Hairer, S.P. Norsett and G. Wanner,
  "Solving Ordinary Differential Equations I: Nonstiff Problems",
  Springer.
  """
  @spec starting_stepsize(
          order :: integer(),
          ode_fn :: RungeKutta.ode_fn_t(),
          t0 :: Nx.t(),
          x0 :: Nx.t(),
          abs_tol :: Nx.t(),
          rel_tol :: Nx.t(),
          norm_control? :: Nx.t()
        ) :: Nx.t()
  defn starting_stepsize(order, ode_fn, t0, x0, abs_tol, rel_tol, norm_control?) do
    nx_type = Nx.type(x0)
    # Compute norm of initial conditions
    x_zeros = zero_vector(Nx.size(x0), Nx.type(x0))
    d0 = Utils.abs_rel_norm(x0, x0, x_zeros, abs_tol, rel_tol, norm_control?)

    x = ode_fn.(t0, x0)

    d1 = Utils.abs_rel_norm(x, x, x_zeros, abs_tol, rel_tol, norm_control?)

    h0 =
      if d0 < 1.0e-5 or d1 < 1.0e-5 do
        Nx.tensor(1.0e-6, type: nx_type)
      else
        Nx.tensor(0.01, type: nx_type) * (d0 / d1)
      end

    # Compute one step of Explicit-Euler
    x1 = x0 + h0 * x

    # Approximate the derivative norm
    xh = ode_fn.(t0 + h0, x1)

    xh_minus_x = xh - x
    d2 = Nx.tensor(1.0, type: nx_type) / h0 * Utils.abs_rel_norm(xh_minus_x, xh_minus_x, x_zeros, abs_tol, rel_tol, norm_control?)

    one = Nx.tensor(1, type: nx_type)

    h1 =
      if max(d1, d2) <= 1.0e-15 do
        max(Nx.tensor(1.0e-06, type: nx_type), h0 * Nx.tensor(1.0e-03, type: nx_type))
      else
        Nx.pow(Nx.tensor(1.0e-02, type: nx_type) / max(d1, d2), one / (order + one))
      end

    min(Nx.tensor(100.0, type: nx_type) * h0, h1)
  end

  @spec convert_to_nx_options(Nx.t() | float(), Nx.t() | float(), integer(), Keyword.t()) :: NxOptions.t()
  deftransform convert_to_nx_options(t_start, t_end, order, opts) do
    nimble_opts = opts |> NimbleOptions.validate!(@options_schema) |> Map.new()
    nx_type = nimble_opts[:type] |> Nx.Type.normalize!()

    max_number_of_errors = nimble_opts[:max_number_of_errors] |> Utils.convert_arg_to_nx_type({:s, 32})

    max_step =
      if max_step = nimble_opts[:max_step] do
        max_step
      else
        default_max_step(t_start, t_end)
      end
      |> Utils.convert_arg_to_nx_type(nx_type)

    fixed_output_times? = nimble_opts[:fixed_output_times?] |> Utils.convert_arg_to_nx_type({:u, 8})

    # If you are using fixed output times, then interpolation is turned off (of course the fixed output
    # point itself is inteprolated):)
    refine = if nimble_opts[:fixed_output_times?], do: 1, else: nimble_opts[:refine]

    fixed_output_step = nimble_opts[:fixed_output_step] |> Utils.convert_arg_to_nx_type(nx_type)
    norm_control? = nimble_opts[:norm_control?] |> Utils.convert_arg_to_nx_type({:u, 8})
    abs_tol = nimble_opts[:abs_tol] |> Utils.convert_arg_to_nx_type(nx_type)
    rel_tol = nimble_opts[:rel_tol] |> Utils.convert_arg_to_nx_type(nx_type)
    nx_while_loop_integration? = nimble_opts[:nx_while_loop_integration?] |> Utils.convert_arg_to_nx_type({:u, 8})

    {speed, nx_while_loop_integration?} =
      case nimble_opts[:speed] do
        :infinite -> {Nx.Constants.infinity(nx_type), nx_while_loop_integration?}
        # If the speed is set to a number other than :infinity, then the Nx `while` loop can no longer be used:
        some_number -> {some_number |> Utils.convert_arg_to_nx_type(nx_type), Nx.u8(0)}
      end

    event_fn_adapter = ExternalFnAdapter.wrap_external_fn_double_arity(nimble_opts[:event_fn])
    zero_fn_adapter = ExternalFnAdapter.wrap_external_fn(nimble_opts[:zero_fn])
    output_fn_adapter = ExternalFnAdapter.wrap_external_fn(nimble_opts[:output_fn])

    non_linear_eqn_root_opt_keys = NonLinearEqnRoot.option_keys()

    non_linear_eqn_root_nx_options =
      opts
      |> Enum.filter(fn {key, _value} -> key in non_linear_eqn_root_opt_keys end)
      |> NonLinearEqnRoot.convert_to_nx_options()

    %NxOptions{
      type: nx_type,
      max_step: max_step,
      refine: refine,
      speed: speed,
      order: order,
      abs_tol: abs_tol,
      rel_tol: rel_tol,
      norm_control?: norm_control?,
      fixed_output_times?: fixed_output_times?,
      fixed_output_step: fixed_output_step,
      max_number_of_errors: max_number_of_errors,
      nx_while_loop_integration?: nx_while_loop_integration?,
      output_fn_adapter: output_fn_adapter,
      event_fn_adapter: event_fn_adapter,
      zero_fn_adapter: zero_fn_adapter,
      non_linear_eqn_root_nx_options: non_linear_eqn_root_nx_options
    }
  end

  # Creates a zero vector that has the length of `x`
  # Is there a better built-in Nx way of doing this?
  @spec zero_vector(Nx.t(), Nx.t()) :: Nx.t()
  defnp zero_vector(size, type) do
    0.0 |> Nx.tensor(type: type) |> Nx.broadcast({size})
  end

  @spec default_max_step(Nx.t(), Nx.t()) :: Nx.t()
  deftransformp default_max_step(%Nx.Tensor{} = t_start, %Nx.Tensor{} = t_end) do
    # See Octave: integrate_adaptive.m:89
    Nx.subtract(t_start, t_end) |> Nx.abs() |> Nx.multiply(Nx.tensor(0.1, type: Nx.type(t_start)))
  end

  @spec default_max_step(Nx.t(), Nx.t()) :: Nx.t()
  deftransformp default_max_step(t_start, t_end) do
    # See Octave: integrate_adaptive.m:89
    0.1 * abs(t_start - t_end)
  end
end
