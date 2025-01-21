defmodule Integrator.MultiIntegrator do
  @moduledoc """
  Integrates multiple simulations that are tied together somehow, such as a bouncing ball
  in the `ballode.m` example.
  """

  alias Integrator.AdaptiveStepsize
  alias Integrator.AdaptiveStepsize.IntegrationStep
  alias Integrator.NonLinearEqnRoot
  alias Integrator.RungeKutta

  @type integration_status :: :halt | :continue | :completed
  @zero_tolerance 1.0e-07
  @type transition_fn_t :: (Nx.t(), Nx.t(), t(), Keyword.t() -> {integration_status, Nx.t(), Nx.t(), Keyword.t()})

  @type t :: %__MODULE__{
          event_t: [Nx.t()],
          event_x: [Nx.t()],
          transition_x: [Nx.t()],
          integrations: [IntegrationStep.t()],
          integration_status: integration_status(),
          t_start: Nx.t(),
          t_end: Nx.t()
        }

  defstruct [
    :t_start,
    :t_end,
    event_t: [],
    event_x: [],
    transition_x: [],
    integrations: [],
    integration_status: :continue
  ]

  all_options =
    NonLinearEqnRoot.options_schema().schema
    |> Keyword.merge(AdaptiveStepsize.options_schema_adaptive_stepsize_only().schema)
    |> Keyword.merge(Integrator.options_schema_integrator_only().schema)

  @options_schema NimbleOptions.new!(all_options)

  @doc """
  Integrates multiple times, with a transition function handling the junction between integrations

  ## Options
  See the options for these functions which are passed through:

  * `Integrator.NonLinearEqnRoot.find_zero/4`
  * `Integrator.AdaptiveStepsize.integrate/9`
  * `Integrator.integrate/4`

  """
  @spec integrate(
          ode_fn :: RungeKutta.ode_fn_t(),
          event_fn :: fun(),
          transition_fn :: transition_fn_t(),
          t_start :: Nx.t(),
          t_end :: Nx.t(),
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: t()
  def integrate(ode_fn, event_fn, transition_fn, t_start, t_end, x0, opts) do
    opts = opts |> NimbleOptions.validate!(@options_schema)
    multi = %__MODULE__{t_start: t_start, t_end: t_end}
    opts = opts |> Keyword.merge(event_fn: event_fn)
    integrate_next_segment(multi, :continue, ode_fn, transition_fn, Nx.to_number(t_start), Nx.to_number(t_end), x0, opts)
  end

  @doc """
  Collates the simulation output from all of the integrations
  """
  @spec all_output_data(t(), atom()) :: [Nx.t()]
  def all_output_data(multi, t_or_x) do
    output =
      multi.integrations
      |> Enum.reverse()
      |> Enum.reduce([], fn integration, acc ->
        [_first | rest] = Map.get(integration, t_or_x)
        [rest | acc]
      end)

    first = multi.integrations |> List.first() |> Map.get(t_or_x) |> List.first()
    [first | output] |> List.flatten()
  end

  # ===========================================================================
  # Private functions below here:

  @spec integrate_next_segment(
          multi :: t(),
          status :: integration_status(),
          ode_fn :: RungeKutta.ode_fn_t(),
          transition_fn :: transition_fn_t(),
          t_start :: float(),
          t_end :: float(),
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: t()
  defp integrate_next_segment(multi, _status, _ode_fn, _transition_fn, t_start, t_end, _x0, _opts)
       when abs(t_start - t_end) < @zero_tolerance or t_start >= t_end do
    %{multi | integration_status: :completed}
    |> reverse_results()
  end

  defp integrate_next_segment(multi, :halt, _ode_fn, _transition_fn, _t_start, _t_end, _x0, _opts) do
    %{multi | integration_status: :halt}
    |> reverse_results()
  end

  defp integrate_next_segment(multi, _status, ode_fn, transition_fn, _t_start, t_end, x0, opts) do
    integration = Integrator.integrate(ode_fn, multi.t_start, multi.t_end, x0, opts)
    new_t_start = integration.t_current
    last_x = integration.x_current
    multi = %{multi | integration_status: integration.terminal_event, integrations: [integration | multi.integrations]}

    {status, new_t_start, new_x0, opts} = transition_fn.(new_t_start, last_x, multi, opts)

    multi = multi |> update(new_t_start, last_x, new_x0)
    integrate_next_segment(multi, status, ode_fn, transition_fn, Nx.to_number(new_t_start), t_end, new_x0, opts)
  end

  @spec reverse_results(t()) :: t()
  defp reverse_results(multi) do
    %{
      multi
      | event_t: Enum.reverse(multi.event_t),
        event_x: Enum.reverse(multi.event_x),
        transition_x: Enum.reverse(multi.transition_x),
        integrations: Enum.reverse(multi.integrations)
    }
  end

  @spec update(t(), Nx.t(), Nx.t(), Nx.t()) :: t()
  defp update(multi, new_t_start, last_x, new_x0) do
    %{
      multi
      | event_t: [new_t_start | multi.event_t],
        event_x: [last_x | multi.event_x],
        transition_x: [new_x0 | multi.transition_x],
        t_start: new_t_start
    }
  end
end
