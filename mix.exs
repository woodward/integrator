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
      #
      # Hex
      description: "Numerical integration in Elixir",
      package: package(),
      elixirc_paths: elixirc_paths(Mix.env()),
      #
      # Docs
      name: "Integrator",
      docs: docs(),
      aliases: aliases()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:csv, "~> 3.0", only: :test},
      {:dialyxir, "~> 1.0", only: :dev, runtime: false},
      {:ex_doc, "~> 0.29", only: :dev, runtime: false},
      {:math, "~> 0.7"},
      {:patch, "~> 0.12.0", only: :test},
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
        "guides/intro_to_integrator.livemd",
        "guides/interpolation_and_fixed_times.livemd",
        "guides/output_functions.livemd",
        "guides/event_functions.livemd",
        "guides/nonlinear_eqn_root.livemd",
        "guides/dormand_prince_45.livemd",
        "guides/bogacki_shampine_23.livemd"
      ],
      before_closing_head_tag: &before_closing_head_tag/1,
      javascript_config_path: nil
    ]
  end

  defp aliases do
    [docs: ["docs", &copy_images/1]]
  end

  defp copy_images(_) do
    File.cp_r("images", "doc/images/")
    # File.cp_r("images", "doc/images/", fn source, destination ->
    #   IO.gets("Overwriting #{destination} by #{source}. Type y to confirm. ") == "y\n"
    # end)
  end

  def before_closing_head_tag(:epub), do: ""
  def before_closing_head_tag(:html), do: File.read!("doc_config/before_closing_head_tag.html")
end
