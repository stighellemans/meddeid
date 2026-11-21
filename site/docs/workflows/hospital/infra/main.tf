terraform {
  required_version = ">= 1.6, < 2.0"
  backend "azurerm" {}
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 4.0" }
  }
}
provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}
variable "subscription_id" { type = string }
variable "resource_group" { type = string }
variable "subnet_id" { type = string }
variable "registry_id" { type = string }
variable "host_image_id" {
  type        = string
  description = "Approved Ubuntu image version with Docker Compose, Azure CLI, and NVIDIA container support."
}
variable "ssh_public_key" { type = string }
variable "vm_name" {
  type    = string
  default = "meddeid-inference"
}
variable "registry_pull_role" {
  type    = string
  default = "AcrPull"
}
data "azurerm_resource_group" "host" { name = var.resource_group }
resource "azurerm_network_interface" "host" {
  name                = "${var.vm_name}-nic"
  location            = data.azurerm_resource_group.host.location
  resource_group_name = data.azurerm_resource_group.host.name
  ip_configuration {
    name                          = "private"
    subnet_id                     = var.subnet_id
    private_ip_address_allocation = "Dynamic"
  }
}
resource "azurerm_linux_virtual_machine" "host" {
  name                            = var.vm_name
  location                        = data.azurerm_resource_group.host.location
  resource_group_name             = data.azurerm_resource_group.host.name
  network_interface_ids           = [azurerm_network_interface.host.id]
  size                            = "Standard_NC4as_T4_v3"
  admin_username                  = "meddeidops"
  disable_password_authentication = true
  source_image_id                 = var.host_image_id
  admin_ssh_key {
    username   = "meddeidops"
    public_key = var.ssh_public_key
  }
  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 256
  }
  identity { type = "SystemAssigned" }
}
resource "azurerm_role_assignment" "pull" {
  scope                = var.registry_id
  role_definition_name = var.registry_pull_role
  principal_id         = azurerm_linux_virtual_machine.host.identity[0].principal_id
  principal_type       = "ServicePrincipal"
}
output "private_ip" { value = azurerm_network_interface.host.private_ip_address }
output "vm_id" { value = azurerm_linux_virtual_machine.host.id }
