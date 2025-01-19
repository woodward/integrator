defmodule Integrator.InternalComputations.IntegrationStepTest do
  @moduledoc false
  use Integrator.TestCase, async: true

  alias Integrator.AdaptiveStepsize.IntegrationStep

  describe "status_integration/1" do
    test "returns :ok if the integration was successful" do
      integration = %IntegrationStep{status_integration: Nx.u8(1)}
      assert IntegrationStep.status_integration(integration) == :ok

      integration = %IntegrationStep{status_integration: Nx.s32(1)}
      assert IntegrationStep.status_integration(integration) == :ok

      integration = %IntegrationStep{status_integration: 1}
      assert IntegrationStep.status_integration(integration) == :ok
    end

    test "returns an error tuple for max errors exceeded" do
      integration = %IntegrationStep{status_integration: IntegrationStep.max_errors_exceeded()}
      assert IntegrationStep.status_integration(integration) == {:error, "Maximum number of errors exceeded"}
    end

    test "returns an error tuple for an unknown error" do
      integration = %IntegrationStep{status_integration: 999}
      assert IntegrationStep.status_integration(integration) == {:error, "Unknown error"}
    end
  end

  describe "status_non_linear_eqn_root/1" do
    test "calls through to the NonLinearEqnRoot functions to get the status" do
      integration = %IntegrationStep{status_non_linear_eqn_root: Nx.u8(2)}
      assert IntegrationStep.status_non_linear_eqn_root(integration) == {:error, "Invalid initial bracket"}
    end
  end
end
