import Config

# if System.get_env("EXLA_ENABLED") == true do
config :integrator, nx_backend: :exla

# config :exla, :add_backend_on_inspect, config_env() != :test

config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  rocm: [platform: :rocm, memory_fraction: 0.8],
  other_host: [platform: :host]

config :exla, default_client: String.to_atom(System.get_env("EXLA_TARGET", "host"))

config :logger, :console,
  format: "\n$time [$level] $metadata $message\n",
  metadata: [:domain, :file, :line]

Nx.global_default_backend(EXLA.Backend)
# end
