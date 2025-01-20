defmodule Integrator.AdaptiveStepsize.NxOptions do
  @moduledoc """
  `NimbleOptions` converted into an Nx-friendly `Nx.Container` struct for use when finding the non-linear eqn root
  (so that the options can be safely passed from Elixir-land to Nx-land).
  """

  alias Integrator.ExternalFnAdapter
  alias Integrator.AdaptiveStepsize
  alias Integrator.NonLinearEqnRoot
  alias Integrator.Utils

  import Nx.Defn

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

  @spec convert_opts_to_nx_options(Nx.t() | float(), Nx.t() | float(), integer(), Keyword.t()) :: NxOptions.t()
  deftransform convert_opts_to_nx_options(t_start, t_end, order, opts) do
    nimble_opts = opts |> NimbleOptions.validate!(AdaptiveStepsize.options_schema()) |> Map.new()
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

    %__MODULE__{
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

  @spec default_max_step(Nx.t() | float(), Nx.t() | float()) :: Nx.t() | float()
  deftransformp default_max_step(%Nx.Tensor{} = t_start, %Nx.Tensor{} = t_end) do
    # See Octave: integrate_adaptive.m:89
    Nx.subtract(t_start, t_end) |> Nx.abs() |> Nx.multiply(Nx.tensor(0.1, type: Nx.type(t_start)))
  end

  deftransformp default_max_step(t_start, t_end) do
    # See Octave: integrate_adaptive.m:89
    0.1 * abs(t_start - t_end)
  end
end
