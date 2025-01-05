defmodule Integrator.AdaptiveStepsizeRefactor do
  @moduledoc """
  Integrates a set of ODEs with an adaptive timestep.
  """

  import Nx.Defn

  alias Integrator.FixedOutputTimes
  alias Integrator.NonLinearEqnRoot
  alias Integrator.Point
  alias Integrator.RungeKutta
  alias Integrator.Utils

  @derive {Nx.Container,
   containers: [
     :t_at_start_of_step,
     :x_at_start_of_step,
     :dt_new,
     :rk_step,
     :fixed_output_times,
     :output_points,
     #
     :count_loop__increment_step,
     :count_cycles__compute_step,
     #
     # ireject in Octave:
     :error_count,
     :i_step,
     #
     :terminal_event,
     :terminal_output,
     #
     :step_start_timestamp_μs,
     :step_elapsed_time_μs,
     #
     :overall_start_timestamp_μs,
     :overall_elapsed_time_μs
   ]}

  @type t :: %__MODULE__{
          t_at_start_of_step: Nx.t(),
          x_at_start_of_step: Nx.t(),
          dt_new: Nx.t(),
          rk_step: RungeKutta.Step.t(),
          fixed_output_times: FixedOutputTimes.t(),
          output_points: {Point.t(), Point.t(), Point.t(), Point.t()},
          #
          count_loop__increment_step: Nx.t(),
          count_cycles__compute_step: Nx.t(),
          #
          # ireject in Octave:
          error_count: Nx.t(),
          i_step: Nx.t(),
          #
          terminal_event: Nx.t(),
          terminal_output: Nx.t(),
          #
          step_start_timestamp_μs: Nx.t(),
          step_elapsed_time_μs: Nx.t(),
          #
          overall_start_timestamp_μs: Nx.t(),
          overall_elapsed_time_μs: Nx.t()
        }
  defstruct [
    :t_at_start_of_step,
    :x_at_start_of_step,
    :dt_new,
    :rk_step,
    :fixed_output_times,
    :output_points,
    #
    :count_loop__increment_step,
    :count_cycles__compute_step,
    #
    # ireject in Octave:
    :error_count,
    :i_step,
    #
    :terminal_event,
    :terminal_output,
    #
    :step_start_timestamp_μs,
    :step_elapsed_time_μs,
    #
    :overall_start_timestamp_μs,
    :overall_elapsed_time_μs
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
    %__MODULE__{t_at_start_of_step: Nx.u8(0)}
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

  # Creates a zero vector that has the length of `x`
  # Is there a better built-in Nx way of doing this?
  @spec zero_vector(Nx.t(), Nx.t()) :: Nx.t()
  defnp zero_vector(size, type) do
    0.0 |> Nx.tensor(type: type) |> Nx.broadcast({size})
  end
end
