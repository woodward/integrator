defmodule Integrator.AdaptiveStepsizeRefactor do
  @moduledoc """
  Integrates a set of ODEs with an adaptive timestep.
  """

  # import Nx.Defn

  alias Integrator.NonLinearEqnRoot
  alias Integrator.RungeKutta

  @derive {Nx.Container,
           containers: [
             :t_old
           ]}

  @type t :: %__MODULE__{
          t_old: Nx.t()
        }
  defstruct [
    :t_old
  ]

  defmodule NxOptions do
    @moduledoc """
    `NimbleOptions` converted into an Nx-friendly `Nx.Container` struct for use when finding the non-linear eqn root
    (so that the options can be safely passed from Elixir-land to Nx-land).
    """

    @derive {Nx.Container,
             containers: [
               :abs_tol,
               :norm_control?,
               :rel_tol
             ],
             keep: []}

    @type t :: %__MODULE__{
            abs_tol: Nx.t(),
            norm_control?: Nx.t(),
            rel_tol: Nx.Type.t()
          }

    defstruct abs_tol: 1.0e-06,
              norm_control?: Nx.u8(1),
              rel_tol: 1.0e-03
  end

  options = [
    abs_tol: [
      type: :any,
      doc: """
      The absolute tolerance used when computing the absolute relative norm. Defaults to 1.0e-06 in the Nx type that's been specified.
      """
    ],
    norm_control: [
      type: :boolean,
      doc: "Indicates whether norm control is to be used when computing the absolute relative norm.",
      default: true
    ],
    rel_tol: [
      type: :any,
      doc: """
       The relative tolerance used when computing the absolute relative norm. Defaults to 1.0e-03 in the Nx type that's been specified.
      """
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
          fixed_times :: [Nx.t()] | nil,
          initial_tstep :: Nx.t(),
          x0 :: Nx.t(),
          order :: integer(),
          opts :: Keyword.t()
        ) :: t()

  def integrate(_stepper_fn, _interpolate_fn, _ode_fn, _t_start, _t_end, _fixed_times, _initial_tstep, _x0, _order, _opts \\ []) do
    %__MODULE__{t_old: Nx.u8(0)}
  end
end
