defmodule Integrator.TestHelpers do
  @moduledoc false
  import ExUnit.Assertions

  @doc """
  Asserts `lhs` is close to `rhs`.
  Copied from [Nx.Helpers.assert_all_close/3](https://github.com/elixir-nx/nx/blob/main/nx/test/support/helpers.ex)
  (which is not included in the released version of Nx, so I cannot just invoke it).
  """
  def assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    equals =
      left
      |> Nx.all_close(right, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if equals != Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend) do
      flunk("""
      expected

      #{inspect(left)}

      to be within tolerance of

      #{inspect(right)}
      """)
    end
  end

  defmacro assert_nx_equal(left, right) do
    # Assert against binary backend tensors to show diff on failure
    quote do
      assert unquote(left) |> to_binary_backend() == unquote(right) |> to_binary_backend()
    end
  end

  defmacro assert_nx_true(tensor) do
    # Assert against binary backend tensors to show diff on failure

    quote do
      assert unquote(tensor) |> to_binary_backend() == Nx.u8(1) |> to_binary_backend()
    end
  end

  defmacro assert_nx_false(tensor) do
    # Assert against binary backend tensors to show diff on failure

    quote do
      assert unquote(tensor) |> to_binary_backend() == Nx.u8(0) |> to_binary_backend()
    end
  end

  def to_binary_backend(tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  @doc """
  Asserts that two lists are equal
  """
  def assert_lists_equal(actual_list, expected_list, delta \\ 0.001) do
    assert length(actual_list) == length(expected_list)

    Enum.zip(actual_list, expected_list)
    |> Enum.map(fn {actual, expected} ->
      assert_in_delta(actual, expected, delta)
    end)
  end

  @doc """
  Asserts that two lists (which contain Nx tensors) are equal
  """
  def assert_nx_lists_equal(actual_list, expected_list, opts \\ []) do
    assert length(actual_list) == length(expected_list)

    Enum.zip(actual_list, expected_list)
    |> Enum.map(fn {actual, expected} ->
      assert_all_close(actual, expected, opts)
    end)
  end

  @doc """
  Asserts that two Nx lists are equal
  """
  def assert_nx_lists_equal_refactor(actual_list, expected_list, opts \\ []) do
    assert length(actual_list) == length(expected_list)

    atol = Keyword.get(opts, :atol, 1.0e-15)
    rtol = Keyword.get(opts, :rtol, 1.0e-15)

    Enum.zip(actual_list, expected_list)
    |> Enum.map(fn {actual, expected} ->
      assert_all_close(actual, expected, atol: atol, rtol: rtol)
    end)
  end

  @doc """
  Reads a CSV file which contains a single column of float values

  Returns a list of float values
  """
  def read_csv(filename) do
    File.stream!(filename)
    |> CSV.decode!(field_transform: &String.trim/1)
    |> Enum.to_list()
    |> List.flatten()
    |> Enum.map(&String.to_float(String.trim(&1)))
  end

  @doc """
  Reads a CSV file and converts the rows into Nx tensors

  Returns a list of Nx tensors
  """
  def read_nx_list(filename) do
    File.stream!(filename)
    |> CSV.decode!(field_transform: &String.trim/1)
    |> Enum.to_list()
    |> Enum.map(fn row ->
      row |> Enum.map(&String.to_float(&1)) |> Nx.tensor(type: :f64)
    end)
  end

  @doc """
  Write the `output_t` values to file. Used when debugging tests
  """
  def write_t(output_t, filename) do
    data = output_t |> Enum.map_join("\n", &Nx.to_number(&1))
    File.write!(filename, data)
  end

  @doc """
  Write the `output_x` values to file. Used when debugging tests
  """
  def write_x(output_x, filename) do
    [first_x | _rest_of_x] = output_x
    {length_of_x} = Nx.shape(first_x)

    data =
      output_x
      |> Enum.map_join("\n", fn x ->
        0..(length_of_x - 1)
        |> Enum.reduce("", fn i, acc ->
          acc <> "#{Nx.to_number(x[i])}  "
        end)
      end)

    File.write!(filename, data)
  end

  def assert_nx_f32(nx_value), do: assert(Nx.type(nx_value) |> Nx.Type.to_string() == "f32")
  def assert_nx_f64(nx_value), do: assert(Nx.type(nx_value) |> Nx.Type.to_string() == "f64")

  def assert_rk_steps_equal(actual, expected) do
    assert_nx_equal(actual.t_old, expected.t_old)
    assert_nx_equal(actual.t_new, expected.t_new)

    assert_nx_equal(actual.x_old, expected.x_old)
    assert_nx_equal(actual.x_new, expected.x_new)

    assert_nx_equal(actual.k_vals, expected.k_vals)
    assert_nx_equal(actual.options_comp, expected.options_comp)
    assert_nx_equal(actual.error_estimate, expected.error_estimate)
    assert_nx_equal(actual.dt, expected.dt)
  end

  def assert_integration_steps_equal(actual, expected) do
    assert_rk_steps_equal(actual.rk_step, expected.rk_step)

    assert_nx_equal(actual.t_current, expected.t_current)
    assert_nx_equal(actual.x_current, expected.x_current)
    assert_nx_equal(actual.dt_new, expected.dt_new)
    assert_nx_equal(actual.dt_new, expected.dt_new)

    assert actual.stepper_fn == expected.stepper_fn
    assert actual.ode_fn == expected.ode_fn
    assert actual.interpolate_fn == expected.interpolate_fn

    assert_nx_equal(actual.fixed_output_t_next, expected.fixed_output_t_next)
    assert_nx_equal(actual.fixed_output_t_within_step?, expected.fixed_output_t_within_step?)

    assert_nx_equal(actual.count_loop__increment_step, expected.count_loop__increment_step)
    assert_nx_equal(actual.count_cycles__compute_step, expected.count_cycles__compute_step)
    assert_nx_equal(actual.error_count, expected.error_count)
    assert_nx_equal(actual.i_step, expected.i_step)

    assert_nx_equal(actual.terminal_event, expected.terminal_event)
    assert_nx_equal(actual.terminal_output, expected.terminal_output)
    assert_nx_equal(actual.status_integration, expected.status_integration)
    assert_nx_equal(actual.status_non_linear_eqn_root, expected.status_non_linear_eqn_root)

    assert_nx_equal(actual.start_timestamp_μs, expected.start_timestamp_μs)
    assert_nx_equal(actual.step_timestamp_μs, expected.step_timestamp_μs)
    assert_nx_equal(actual.elapsed_time_μs, expected.elapsed_time_μs)
  end
end
