# Add the first MedDeID GPU host

Use this recipe if the hospital already manages Azure with Terraform. The
Databricks job needs a private address for the model, and the model needs a GPU
and access to its software. Your existing resource group, services subnet and
container registry provide the rest.

Add the blocks below to your existing root configuration, for example in
`meddeid.tf`. Terraform will use the project's provider, state and deployment
identity. Replace the example `data.azurerm_…` references with the resource,
data source or module outputs your project uses. Follow the hospital's naming
and tagging convention. The examples use `acc` for the acceptance environment.

## Give the host a private address

Put the network interface on the existing services subnet so the job can reach
the model over the hospital network. With `var.environment = "acc"`, this
creates `nic-meddeid-acc-01`.

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
```

## Give the model a GPU and a persistent disk

The T4 runs detection. The managed OS disk keeps the installation when the VM
is deallocated between jobs. In acceptance, these resources are named
`vm-meddeid-acc-01` and `osdisk-meddeid-acc-01`. Supply your SSH public key
and the version ID of an approved Ubuntu image with Docker Compose, an NVIDIA
driver, NVIDIA Container Toolkit and Azure CLI.

Use a 64 GiB OS disk for a new T4 inference host that pulls prebuilt images.
In the Ubuntu 24.04 test with the public Dutch model, at least 23.7 GiB
remained free during the upgrade from 0.4.2 to 0.4.3rc4 and rollback.
The build cache and test registry used a separate disk. Check that the
hospital’s base image fits the requested disk size. For an existing VM with a
larger OS disk, keep its current size in Terraform.

```hcl
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
    disk_size_gb         = 64
  }

  identity { type = "SystemAssigned" }
}
```

## Let the VM pull the approved images

The VM uses its own system-assigned identity to read the hospital's container
registry. The Databricks job has a different identity. It needs permission to
call the HTTPS service and, for keyed replacements, to read the selected
mapping key from Key Vault.

```hcl
resource "azurerm_role_assignment" "meddeid_registry_pull" {
  scope                = data.azurerm_container_registry.platform.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_linux_virtual_machine.meddeid.identity[0].principal_id
  principal_type       = "ServicePrincipal"
}
```

For a registry using ABAC repository permissions, use
`Container Registry Repository Reader` for the pull assignment.
[Azure registry roles for a managed identity](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication-managed-identity)

## Give Databricks a name it can connect to

Register the private IP in the hospital's existing DNS zone. For example, the
job could connect to `meddeid-acc.clinical.example.org`. Issue the TLS
certificate for that name and allow TCP 443 from approved job subnets. Keep VM
management on the hospital's private management route.

If this project manages the DNS zone, add the A record here using the zone's
existing name and resource group.

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

Review the hospital's normal Terraform plan and apply it through the project's
usual process. Once the VM is up, check GPU access on the host and inside an
approved CUDA container. Stage the verified model bundle and continue with
[Start the HTTPS service](deploy-service.md#start-the-https-service).
