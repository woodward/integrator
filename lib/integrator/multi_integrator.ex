defmodule Integrator.MultiIntegrator do
  @moduledoc false

  alias Integrator.AdaptiveStepsize

  @type integration_status :: :halt | :continue | :completed
  @zero_tolerance 1.0e-07

  @type t :: %__MODULE__{
          event_t: [Nx.t()],
          event_x: [Nx.t()],
          transition_x: [Nx.t()],
          integrations: [AdaptiveStepsize.t()],
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

  @spec integrate(
          ode_fn :: fun(),
          event_fn :: fun(),
          transition_fn :: fun(),
          t_start :: Nx.t(),
          t_end :: Nx.t(),
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: t()
  def integrate(ode_fn, event_fn, transition_fn, t_start, t_end, x0, opts) do
    multi = %__MODULE__{t_start: t_start, t_end: t_end}
    opts = opts |> Keyword.merge(event_fn: event_fn)
    integrate_next_segment(multi, :continue, ode_fn, transition_fn, Nx.to_number(t_start), Nx.to_number(t_end), x0, opts)
  end

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
          ode_fn :: fun(),
          transition_fn :: fun(),
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

  # defp integrate_next_segment(multi, :halt, _ode_fn, _transition_fn, _t_start, _t_end, _x0, _opts) do
  #   %{multi | integration_status: :halt}
  #   |> reverse_results()
  # end

  defp integrate_next_segment(multi, _status, ode_fn, transition_fn, _t_start, t_end, x0, opts) do
    integration = Integrator.integrate(ode_fn, [multi.t_start, multi.t_end], x0, opts)
    new_t_start = List.last(integration.output_t)
    last_x = List.last(integration.output_x)

    new_x0 = transition_fn.(new_t_start, last_x)

    multi = multi |> update(new_t_start, last_x, new_x0, integration, integration.terminal_event)
    integrate_next_segment(multi, :continue, ode_fn, transition_fn, Nx.to_number(new_t_start), t_end, new_x0, opts)
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

  @spec update(t(), Nx.t(), Nx.t(), Nx.t(), AdaptiveStepsize.t(), integration_status()) :: t()
  defp update(multi, new_t_start, last_x, new_x0, integration, integration_event) do
    %{
      multi
      | integration_status: integration_event,
        integrations: [integration | multi.integrations],
        event_t: [new_t_start | multi.event_t],
        event_x: [last_x | multi.event_x],
        transition_x: [new_x0 | multi.transition_x],
        t_start: new_t_start
    }
  end
end
