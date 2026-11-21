# Add the first MedDeID GPU host

Use this recipe when the hospital already manages Azure with Terraform and is adding its first MedDeID service. Add the HCL below to the existing root configuration, for example in `meddeid.tf`. It uses the project's provider, state and deployment identity.

For an acceptance environment (`var.environment = "acc"`), Terraform creates the following resources.

| Resource | Example Azure name | Purpose |
| --- | --- | --- |
| Network interface | `nic-meddeid-acc-01` | Private IP on the existing services subnet |
| T4 VM and managed OS disk | `vm-meddeid-acc-01`, `osdisk-meddeid-acc-01` | Run the inference container, keep its installation across restarts |
| Registry role assignment | Azure assigns the role assignment ID | Allow the VM's system-assigned identity to pull the approved images |

Replace the three `data.azurerm_…` references with the resource, data source or module outputs your project already uses for the resource group, subnet and registry. Use your existing environment code, SSH public key and approved Ubuntu GPU image version ID. That image must provide Docker Compose, an NVIDIA driver, NVIDIA Container Toolkit and Azure CLI. Apply the hospital's naming and tagging convention to these examples.

```hcl
locals {
  meddeid_host = "meddeid-${var.environment}-01"
  meddeid_tags = { workload = "meddeid", environment = var.environment }
}

resource "azurerm_network_interface" "meddeid" {
  name                = "nic-${local.meddeid_host}"
  resource_group_name = data.azurerm_resource_group.platform.name
  location            = data.azurerm_resource_group.platform.location
  tags                = local.meddeid_tags

  ip_configuration {
    name                          = "private"
    subnet_id                     = data.azurerm_subnet.services.id
    private_ip_address_allocation = "Dynamic"
  }
}

resource "azurerm_linux_virtual_machine" "meddeid" {
  name                            = "vm-${local.meddeid_host}"
  resource_group_name             = data.azurerm_resource_group.platform.name
  location                        = data.azurerm_resource_group.platform.location
  size                            = "Standard_NC4as_T4_v3"
  network_interface_ids           = [azurerm_network_interface.meddeid.id]
  source_image_id                 = var.host_image_id
  admin_username                  = "meddeidops"
  disable_password_authentication = true
  tags                            = local.meddeid_tags

  admin_ssh_key {
    username   = "meddeidops"
    public_key = var.ssh_public_key
  }

  os_disk {
    name                 = "osdisk-${local.meddeid_host}"
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 256
  }

  identity { type = "SystemAssigned" }
}

resource "azurerm_role_assignment" "meddeid_registry_pull" {
  scope                = data.azurerm_container_registry.platform.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_virtual_machine.meddeid.identity[0].principal_id
  principal_type       = "ServicePrincipal"
}
```

For a registry using ABAC repository permissions, use `Container Registry Repository Reader` for the pull assignment. [Azure registry roles for a managed identity](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication-managed-identity)

**Connect the job network.** Register a private DNS name such as `meddeid-acc.clinical.example.org` pointing to `azurerm_network_interface.meddeid.private_ip_address`. Use the hospital's existing private DNS zone, allow TCP 443 from approved job subnets, and issue the TLS certificate for that name. Keep VM management on the hospital's private management route.

If this Terraform project manages the private DNS zone, add its A record here. Use the existing zone and its resource group.

```hcl
resource "azurerm_private_dns_a_record" "meddeid" {
  name                = "meddeid-${var.environment}"
  zone_name           = data.azurerm_private_dns_zone.clinical.name
  resource_group_name = data.azurerm_private_dns_zone.clinical.resource_group_name
  ttl                 = 300
  records             = [azurerm_network_interface.meddeid.private_ip_address]
  tags                = local.meddeid_tags
}
```

**Connect the application.** The release owner supplies the model bundle and checksum through the agreed storage location. The Databricks job uses its own identity to call the HTTPS endpoint and read the approved mapping-key version from Key Vault. These permissions are separate from the VM's registry access.

Review the additions with the hospital's normal Terraform plan. After deployment, verify GPU access on the VM and continue with [Start the HTTPS service](index.md#3-start-the-service-behind-nginx).
