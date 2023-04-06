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
      {:ex_doc, "~> 0.29", only: :dev, runtime: false},
      {:nx, "~> 0.5"}
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
      before_closing_head_tag: &katex_js_and_css_files/1,
      before_closing_body_tag: &katex_config_js/1,
      javascript_config_path: "../docs_config.js"
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

  defp katex_config_js(:epub), do: nil
  defp katex_config_js(:html), do: ~S(<script src="../katex-config.js"></script>)

  defp katex_js_and_css_files(:epub), do: nil

  defp katex_js_and_css_files(:html) do
    """
    <!--  ----------------------------------------- -->
    <!-- Taken from: view-source:https://katex.org/ -->

    <link rel="preload" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/fonts/KaTeX_Main-Regular.woff2" as="font" type="font/woff2" crossorigin="anonymous">
    <link rel="preload" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/fonts/KaTeX_Math-Italic.woff2" as="font" type="font/woff2" crossorigin="anonymous">
    <link rel="preload" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/fonts/KaTeX_Size2-Regular.woff2" as="font" type="font/woff2" crossorigin="anonymous">
    <link rel="preload" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/fonts/KaTeX_Size4-Regular.woff2" as="font" type="font/woff2" crossorigin="anonymous">

    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lato:300,400,700,700i">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css" crossorigin="anonymous">
    <link rel="stylesheet" href="static/index.css">

    <!--  the "defer" attribute that is on view-source:https://katex.org/ causes KaTeX to not work, so removing it here -->
    <script src="https://cdn.jsdelivr.net/npm/clipboard@2.0.1/dist/clipboard.min.js" integrity="sha256-hIvIxeqhGZF+VVeM55k0mJvWpQ6gTkWk3Emc+NmowYA=" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/mhchem.min.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/copy-tex.min.js" crossorigin="anonymous"></script>
    <script src="js/index.js" type="text/javascript"></script>
    <script src="https://cdn.jsdelivr.net/npm/webfontloader@1.6.28/webfontloader.js" integrity="sha256-4O4pS1SH31ZqrSO2A/2QJTVjTPqVe+jnYgOWUVr7EEc=" crossorigin="anonymous"></script>

    <!--  ----------------------------------------- -->
    """
  end
end
