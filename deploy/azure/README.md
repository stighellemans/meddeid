# Ephemeral Azure T4 release runner

These scripts prepare the one-job GitHub Actions runner required by the CUDA
and T4 TensorRT release gates. They are maintainer infrastructure, not part of
an end-user image. At the time of release preparation no Azure VM or repository
runner existed, so `t4_release_runner_ready` remains a pending suite gate.

Create a new `Standard_NC4as_T4_v3` VM from a supported Ubuntu 24.04 image with
Standard security (Secure Boot and vTPM disabled), SSH-key authentication, and
at least a 128 GB OS disk for the transient builder layers. Follow Microsoft's supported N-series driver
procedure; do not substitute a CUDA toolkit for the host driver.

For example, after selecting a region where the subscription has NCasT4_v3
quota:

```bash
az group create --name meddeid-release-t4 --location REGION
az vm create \
  --resource-group meddeid-release-t4 \
  --name meddeid-t4-runner \
  --location REGION \
  --image Ubuntu2404 \
  --size Standard_NC4as_T4_v3 \
  --security-type Standard \
  --admin-username azureuser \
  --generate-ssh-keys \
  --os-disk-size-gb 128 \
  --public-ip-sku Standard
```

Install the supported NVIDIA driver, reboot if its installer requires it, copy
`bootstrap_t4_runner_host.sh` to the VM, and run it as root. The bootstrap pins
and verifies GitHub Actions Runner 2.337.0, installs Docker Buildx/Compose and
NVIDIA Container Toolkit, validates GPU passthrough, and creates an
unprivileged `actions` user. It never receives a GitHub token.

On the maintainer machine, generate a one-job repository JIT configuration:

```bash
./deploy/azure/create_jit_config.sh /tmp/meddeid-t4.jit
scp /tmp/meddeid-t4.jit azureuser@VM:/tmp/meddeid-t4.jit
rm /tmp/meddeid-t4.jit
```

Move it to the `actions` user with mode 600, then start the one-job runner:

```bash
sudo install -o actions -g actions -m 600 \
  /tmp/meddeid-t4.jit /home/actions/meddeid-t4.jit
sudo rm /tmp/meddeid-t4.jit
sudo -u actions \
  /path/to/run_ephemeral_runner.sh /home/actions/meddeid-t4.jit
```

The runner deletes the JIT file before accepting work and deregisters after one
job. Before every registration, the launcher verifies the dedicated-host
sentinel and removes the prior Actions workspace plus all Docker containers,
images, volumes, and builder cache. A separate JIT configuration is therefore
required for each of the manual CUDA candidate run, manual TensorRT candidate
run, CUDA tag run, and TensorRT tag run. The two tag jobs may queue; start the
next one-job registration after the preceding runner exits. Retain Actions
logs/evidence externally before that cleanup, then delete the entire resource
group to stop billing:

```bash
az group delete --name meddeid-release-t4
```

Resolve and inspect the exact resource group before running that destructive
command. Never place patient data, long-lived cloud credentials, or reusable
GitHub tokens on the runner.
