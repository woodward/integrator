import Config

os_type = :os.type()
apple_arm64? = :erlang.system_info(:system_architecture) |> List.starts_with?(~c"aarch64")

case System.get_env("NX_BACKEND", "BINARY") do
  # ------------------------------------------------------------------------------------------------
  "EXLA" ->
    # config :exla, :add_backend_on_inspect, config_env() != :test

    config :exla, :clients,
      cuda: [platform: :cuda, memory_fraction: 0.8],
      rocm: [platform: :rocm, memory_fraction: 0.8],
      other_host: [platform: :host]

    config :exla,
      default_client: String.to_atom(System.get_env("EXLA_TARGET", "host")),
      is_mac_arm: apple_arm64?

    config :logger, :console,
      format: "\n$time [$level] $metadata $message\n",
      metadata: [:domain, :file, :line]

    Nx.global_default_backend(EXLA.Backend)
    Nx.Defn.global_default_options(compiler: EXLA)

  # ------------------------------------------------------------------------------------------------
  "TORCHX" ->
    config :torchx,
      add_backend_on_inspect: config_env() != :test,
      check_shape_and_type: config_env() == :test,
      is_apple_arm64: apple_arm64?

  # ------------------------------------------------------------------------------------------------
  "BINARY" ->
    Nx.global_default_backend(Nx.BinaryBackend)
end
