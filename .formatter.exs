# Used by "mix format"
locals_without_parens = [
  # assert_nx_equal: 2,
  assert_nx_true: 1,
  assert_nx_false: 1,
  assert_nx_f32: 1,
  assert_nx_f64: 1
]

[
  import_deps: [:nx],
  inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}"],
  line_length: 130,
  locals_without_parens: locals_without_parens,
  export: [locals_without_parens: locals_without_parens]
]
