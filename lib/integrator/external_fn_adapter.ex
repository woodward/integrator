defmodule Integrator.ExternalFnAdapter do
  @moduledoc """
  An adapter which wraps an external, non-defn (i.e., Elixir) function (such as for generating side
  effects, like output) in an Nx.Container struct so that it can be passed as an argument to functions.
  It also encapsulutes the Nx [hook/3](https://hexdocs.pm/nx/Nx.Defn.Kernel.html#hook/3) logic in one spot.
  """

  import Nx.Defn

  @derive {Nx.Container,
           containers: [],
           keep: [
             :external_fn
           ]}

  @type t :: %__MODULE__{
          external_fn: fun()
        }

  @spec no_op_fn(any()) :: any()
  defn no_op_fn(arg), do: arg

  # Note that just doing &(&1) did not work here; the no_op_fn/1 had to be defined above instead
  defstruct external_fn: &__MODULE__.no_op_fn/1

  @doc """
  Invoke an external (i.e., Elixir) function via an Nx hook
  """
  @spec invoke_external_fn(Nx.t(), Nx.t()) :: Nx.t()
  defn invoke_external_fn(z, external_fn_adapter) do
    {z, _external_fn_adapter} =
      hook({z, external_fn_adapter}, fn {zz, adapter} ->
        adapter.external_fn.(zz)
        {zz, adapter}
      end)

    z
  end

  deftransform wrap_external_fn(nil = _external_fn), do: %__MODULE__{}
  deftransform wrap_external_fn(external_fn), do: %__MODULE__{external_fn: external_fn}
end
