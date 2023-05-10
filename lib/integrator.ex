defmodule Integrator do
  @moduledoc """
  $$  F = M A $$

  or an inline equation via a single dollar sign, such as $ F = M A $

  Some more equations:

  $$ f(x) =  \\int_{-\\infty}^\\infty
    f\\hat(\\xi)\\,e^{2 \\pi i \\xi x}
    \\,d\\xi $$

  """

  alias Integrator.{AdaptiveStepsize, Utils}
  alias Integrator.RungeKutta.{BogackiShampine23, DormandPrince45}

  @integrator_options %{
    ode45: DormandPrince45,
    ode23: BogackiShampine23
  }

  @default_refine_opts %{
    ode45: 4,
    ode23: 1
  }

  @default_opts [
    integrator: :ode45
  ]

  defmodule ArgPrecisionError do
    defexception message: "argument precision error",
                 invalid_argument: nil,
                 argument_name: nil,
                 expected_precision: nil,
                 actual_precision: nil
  end

  @doc """
  Integrates an ODE function using either the Dormand-Prince45 method or the Bogacki-Shampine23 method
  """
  @spec integrate(
          ode_fn :: fun(),
          t_start_t_end :: Nx.t() | [float() | Nx.t()],
          x0 :: Nx.t(),
          opts :: Keyword.t()
        ) :: AdaptiveStepsize.t()
  def integrate(ode_fn, t_start_t_end, x0, opts \\ []) do
    opts = (@default_opts ++ AdaptiveStepsize.abs_rel_norm_opts()) |> Keyword.merge(opts) |> set_default_refine_opt()
    opts = opts |> Keyword.put_new_lazy(:type, fn -> Utils.type_atom(x0) end)

    integrator_mod =
      Map.get_lazy(@integrator_options, opts[:integrator], fn ->
        raise "Currently only DormandPrince45 (ode45) and BogackiShampine23 (ode23) are supported"
      end)

    order = integrator_mod.order()
    {t_start, t_end, fixed_times} = parse_start_end(t_start_t_end, opts[:type])

    initial_step =
      Keyword.get_lazy(opts, :initial_step, fn ->
        AdaptiveStepsize.starting_stepsize(order, ode_fn, t_start, x0, opts[:abs_tol], opts[:rel_tol], opts)
      end)

    validiate_args_precision(
      [
        x0: x0,
        initial_step: initial_step,
        abs_tol: opts[:abs_tol],
        rel_tol: opts[:rel_tol]
        # max_step: opts[:max_step]
      ],
      opts[:type]
    )

    AdaptiveStepsize.integrate(
      &integrator_mod.integrate/5,
      &integrator_mod.interpolate/4,
      ode_fn,
      t_start,
      t_end,
      fixed_times,
      initial_step,
      x0,
      order,
      opts
    )
  end

  # @spec parse_start_end([float() | Nx.t()] | Nx.t(), Nx.Type.t()) :: {Nx.t(), Nx.t(), [Nx.t()] | nil}
  defp parse_start_end([t_start, t_end], nx_type) do
    validiate_args_precision([t_start: t_start, t_end: t_end], nx_type)
    {Nx.tensor(t_start, type: nx_type), Nx.tensor(t_end, type: nx_type), _fixed_times = nil}
  end

  defp parse_start_end(t_range, nx_type) do
    validiate_args_precision([t_range: t_range], nx_type)

    t_start = t_range[0]
    {length} = Nx.shape(t_range)

    t_end = t_range[length - 1]

    # Figure out the correct way to do this; there's got to be a better way!
    fixed_times =
      0..(length - 1)
      |> Enum.reduce([], fn i, acc ->
        [t_range[i] | acc]
      end)
      |> Enum.reverse()

    {t_start, t_end, fixed_times}
  end

  @spec set_default_refine_opt(Keyword.t()) :: Keyword.t()
  defp set_default_refine_opt(opts) do
    if opts[:refine] do
      opts
    else
      default_refine_for_integrator = Map.get(@default_refine_opts, opts[:integrator])
      opts |> Keyword.merge(refine: default_refine_for_integrator)
    end
  end

  # @spec validiate_args_precision(Keyword.t(), atom()) :: atom()
  defp validiate_args_precision(args, expected_nx_type) do
    args
    |> Enum.each(fn {arg_name, arg_value} ->
      nx_type = Utils.nx_type_atom(arg_value)

      if nx_type != expected_nx_type do
        raise ArgPrecisionError,
          invalid_argument: arg_value,
          expected_precision: expected_nx_type,
          actual_precision: nx_type,
          argument_name: arg_name
      end
    end)

    :ok
  end
end
