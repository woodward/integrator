defmodule Integrator.MixProject do
  @moduledoc false
  use Mix.Project

  @source_url "https://github.com/woodward/integrator"
  @version "0.1.0"

  def project do
    [
      app: :integrator,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      # Hex
      description: "Numerical integration in Elixir",
      package: package(),
      elixirc_paths: elixirc_paths(Mix.env()),
      #
      # Docs
      name: "Integrator",
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.0", only: [:dev], runtime: false},
      #
      {:ex_doc, github: "elixir-lang/ex_doc", commit: "2c581239e0989fdc91e3c83b4ce28ef4fe0adda6", only: :dev, runtime: false},
      # {:ex_doc, "~> 0.29", only: :dev, runtime: false},
      {:nx, "~> 0.5"}
    ]
  end

  defp package do
    [
      maintainers: ["Greg Woodward"],
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(README.md lib)
    ]
  end

  defp docs() do
    [
      main: "readme",
      source_url: @source_url,
      extras: [
        "README.md",
        "guides/intro-to-integrator.livemd",
        "guides/examples-of-usage.livemd"
      ],
      before_closing_head_tag: &before_closing_head_tag/1,
      javascript_config_path: nil
    ]
  end

  def before_closing_head_tag(:epub), do: ""

  def before_closing_head_tag(:html) do
    # File.cp("doc_config/docs_config.js", "doc")
    # File.cp("doc_config/katex_config.js", "doc")

    File.read!("doc_config/before_closing_head_tag.html")
  end

  # def before_closing_head_tag(:html), do: ""

  def before_closing_body_tag(:epub), do: ""
  def before_closing_body_tag(:html), do: File.read!("doc_config/before_closing_body_tag.html")
  # def before_closing_body_tag(:html), do: ""
end
