defmodule Integrator.Helpers do
  @moduledoc false
  import ExUnit.Assertions

  @doc """
  Asserts `lhs` is close to `rhs`.  Copied from Nx.Helpers (which is not in the released version of Nx)

  https://github.com/elixir-nx/nx/blob/main/nx/test/support/helpers.ex
  """
  def assert_all_close(lhs, rhs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    unless Nx.all_close(lhs, rhs, atol: atol, rtol: rtol, equal_nan: opts[:equal_nan]) ==
             Nx.tensor(1, type: {:u, 8}) do
      flunk("""
      expected

      #{inspect(lhs)}

      to be within tolerance of

      #{inspect(rhs)}
      """)
    end
  end

  def assert_lists_equal(actual_list, expected_list, delta \\ 0.001) do
    assert length(actual_list) == length(expected_list)

    Enum.zip(actual_list, expected_list)
    |> Enum.map(fn {actual, expected} ->
      assert_in_delta(actual, expected, delta)
    end)
  end

  def assert_nx_lists_equal(actual_list, expected_list, opts \\ []) do
    assert length(actual_list) == length(expected_list)

    Enum.zip(actual_list, expected_list)
    |> Enum.map(fn {actual, expected} ->
      assert_all_close(actual, expected, opts)
    end)
  end

  def van_der_pol_fn(_t, y) do
    # https://octave.sourceforge.io/octave/function/ode45.html
    # From octave with y(1) => y(0) and y(2) => y(1):
    # fvdp = @(t,y) [y(1); (1 - y(0)^2) * y(1) - y(0)];
    y0 = y[0]
    y1 = y[1]

    one = Nx.tensor(1.0, type: :f32)
    new_y1 = Nx.subtract(one, Nx.pow(y0, 2)) |> Nx.multiply(y1) |> Nx.subtract(y0)
    Nx.stack([y1, new_y1])
  end
end
