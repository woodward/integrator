#!/usr/bin/env elixir

Mix.install([{:nx, "~> 0.9"}])

import IO.ANSI

defmodule PrecisionQuestions do
  import Nx.Defn

  defn x_times_point_seven_returns_f64_precision_result(x) do
    point_seven = Nx.tensor(0.7, type: Nx.type(x))
    {print_expr(x * point_seven), point_seven}
  end

  defn x_times_point_seven_returns_f32_precision_result(x) do
    point_seven = 0.7
    {print_expr(x * point_seven), point_seven}
  end

  defn x_times_point_five_returns_f64_precision_result(x) do
    point_five = 0.5
    {x * point_five, point_five}
  end
end

# ------------------------------------------------------------------------------------

two_f64 = Nx.tensor(2, type: :f64)

IO.puts(light_yellow() <> "==================================================================" <> reset())
dbg(two_f64)

# two_f64 #=> #Nx.Tensor<
#   f64
#   2.0
# >

IO.puts(light_green() <> "==================================================================" <> reset())
IO.puts(light_yellow() <> ":f64 - 2 * 0.7" <> reset())

{result_with_f64_precision, point_seven_f64} = PrecisionQuestions.x_times_point_seven_returns_f64_precision_result(two_f64)
dbg(result_with_f64_precision)
dbg(point_seven_f64)

# This shows :f64 precision
# result_with_f64_precision #=> #Nx.Tensor<
#   f64
#   1.4
# >

# [show_precision_issue.exs:41: (file)]
# point_seven_f64 #=> #Nx.Tensor<
#   f64
#   0.7
# >

IO.puts(
  light_yellow() <> "Result of Nx.all_close(result_with_f64_precision, Nx.f64(1.4), atol: 1.0e-14, rtol: 1.0e-14):  " <> reset()
)

IO.puts(
  light_green() <> inspect(Nx.all_close(result_with_f64_precision, Nx.f64(1.4), atol: 1.0e-14, rtol: 1.0e-14)) <> "\n" <> reset()
)

IO.puts(light_green() <> "==================================================================" <> reset())
IO.puts(light_yellow() <> ":f32 - 2 * 0.7" <> reset())

# result_with_f32_precision #=> #Nx.Tensor<
#   f64
#   1.399999976158142
# >

# Note how 0.7 is :f32 and has :f32 precision (7 digits of precision rather than 14)
# [show_precision_issue.exs:61: (file)]
# point_seven_f32 #=> #Nx.Tensor<
#   f32
#   0.699999988079071
# >

{result_with_f32_precision, point_seven_f32} = PrecisionQuestions.x_times_point_seven_returns_f32_precision_result(two_f64)
dbg(result_with_f32_precision)

IO.puts(light_red() <> "Note how 0.7 is :f32 and has :f32 precision (7 digits of precision rather than 14)" <> reset())
dbg(point_seven_f32)

# This shows :f32 precision (note that 0.7 is interpreted as :f32):

IO.puts(
  light_yellow() <> "Result of Nx.all_close(result_with_f32_precision, Nx.f64(1.4), atol: 1.0e-14, rtol: 1.0e-14):  " <> reset()
)

IO.puts(
  light_red() <> inspect(Nx.all_close(result_with_f32_precision, Nx.f64(1.4), atol: 1.0e-14, rtol: 1.0e-14)) <> "\n" <> reset()
)

IO.puts(
  light_yellow() <> "Result of Nx.all_close(result_with_f32_precision, Nx.f64(1.4), atol: 1.0e-07, rtol: 1.0e-07):  " <> reset()
)

IO.puts(
  light_green() <> inspect(Nx.all_close(result_with_f32_precision, Nx.f64(1.4), atol: 1.0e-07, rtol: 1.0e-07)) <> "\n" <> reset()
)


IO.puts(light_green() <> "==================================================================" <> reset())
IO.puts(light_yellow() <> ":f64 - 2 * 0.5" <> reset())

{second_result_with_f64_precision, point_five_f32} = PrecisionQuestions.x_times_point_five_returns_f64_precision_result(two_f64)
dbg(second_result_with_f64_precision)
dbg(point_five_f32)

# second_result_with_f64_precision #=> #Nx.Tensor<
#   f64
#   1.0
# >

# point_five_f32 #=> #Nx.Tensor<
#   f32
#   0.5
# >
