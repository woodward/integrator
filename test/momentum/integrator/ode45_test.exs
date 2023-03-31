defmodule Momentum.Integrator.Ode45Test do
  @moduledoc false
  use Momentum.TestCase
  alias Momentum.Integrator.Ode45

  describe "overall" do
    test "van_der_pol_fn" do
      x = Nx.tensor(0.0341, type: :f32)
      y = Nx.tensor([1.9975, -0.0947], type: :f32)

      # From octave with y(1) -> y(0) and y(2) -> y(1):
      # fvdp = @(t,y) [y(1); (1 - y(0)^2) * y(1) - y(0)];
      y0 = 1.9975
      y1 = -0.0947
      expected_y0 = y1
      expected_y1 = (1.0 - y0 * y0) * y1 - y0
      assert expected_y0 == -0.0947
      assert expected_y1 == -1.714346408125

      y_result = van_der_pol_fn(x, y)
      expected_y_result = Nx.tensor([expected_y0, expected_y1])
      assert_all_close(y_result, expected_y_result)
    end

    @tag :skip
    test "performs the integration" do
      # See:
      # https://octave.sourceforge.io/octave/function/ode45.html
      #
      # fvdp = @(t,y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
      # [t,y] = ode45 (fvdp, [0, 20], [2, 0]);

      initial_y = [2.0, 0.0]
      t_initial = 0.0
      t_final = 20.0
      [t, y] = Ode45.integrate(&van_der_pol_fn/2, t_initial, t_final, initial_y)

      expected_t =
        File.read!("test/fixtures/momentum/integrator/runge_kutta_45_test/time.csv")
        |> String.split("\n", trim: true)
        |> Enum.map(&String.to_float(String.trim(&1)))

      expected_y = read_nx_list("test/fixtures/momentum/integrator/runge_kutta_45_test/y.csv")

      assert_lists_equal(t, expected_t)
      assert_nx_lists_equal(y, expected_y)
    end
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

  defp read_nx_list(filename) do
    File.read!(filename)
    |> String.split("\n", trim: true)
    |> Enum.map(fn line ->
      [y0, y1] = String.split(line, ",")
      Nx.tensor([String.to_float(String.trim(y0)), String.to_float(String.trim(y1))])
    end)
  end
end
