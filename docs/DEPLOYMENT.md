# Deployment, security and publishing

KernelLoom is designed first for local and private-network execution. This
guide covers service boundaries, environment settings, operational checks and
the repository's automatic PyPI workflow.

## Local service

The safest default starts on loopback only:

```bash
kernelloom serve --host 127.0.0.1 --port 11435
```

Confirm health:

```bash
curl http://127.0.0.1:11435/health
```

Expected response:

```json
{"status":"ok","models":0}
```

The model count increases after a successful load.

## Network service

Before binding to `0.0.0.0`:

1. Set a long random `KERNELLOOM_API_KEY`.
2. Place the service behind TLS when traffic leaves the machine.
3. Restrict inbound connections with a host or network firewall.
4. Do not give untrusted users access to `/v1/models/load`.
5. Run the process as an account with access only to intended model paths.
6. Set memory and process limits appropriate for the host.

Example:

```powershell
$env:KERNELLOOM_API_KEY = "replace-with-a-long-random-value"
kernelloom serve --host 0.0.0.0 --port 11435
```

KernelLoom provides bearer-token checking, not user management, TLS, rate
limiting or tenant isolation. Add those controls at a trusted reverse proxy or
application gateway when required.

## Native worker boundary

OpenVINO execution runs in a child Python process. The host and worker exchange
JSON Lines through inherited stdin/stdout pipes. The worker does not create a
listening socket.

Set its interpreter explicitly:

```powershell
$env:KERNELLOOM_ACCELERATOR_PYTHON = "D:\runtimes\openvino\Scripts\python.exe"
```

The worker owns native device handles and resident OpenVINO models. Native
generation requests are not automatically replayed after worker failure.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `KERNELLOOM_API_KEY` | Require bearer authentication on `/v1` routes. |
| `KERNELLOOM_DATA_DIR` | Runtime directory used by server-side hardware discovery when no model-specific directory is selected. |
| `KERNELLOOM_ACCELERATOR_PYTHON` | Python interpreter used for the isolated OpenVINO worker. |
| `KERNELLOOM_SOURCE_ROOT` | Explicit source root made available to the worker. |
| `KERNELLOOM_LLAMA_SERVER` | Optional llama.cpp server executable used during hardware discovery. |
| `KERNELLOOM_MEMORY_BANDWIDTH_GBPS` | Override estimated CPU memory bandwidth. |
| `KERNELLOOM_GPU_BANDWIDTH_GBPS` | Override estimated GPU memory bandwidth. |
| `KERNELLOOM_NPU_BANDWIDTH_GBPS` | Override estimated NPU memory bandwidth. |

Bandwidth overrides affect analytical estimates. They are not measurements.

## Data and caches

The default runtime directory is `~/.kernelloom`. A configured directory may
contain:

- `kernelloom.sqlite3` for plans, evidence, roles and audit events;
- compiled OpenVINO caches;
- runtime state needed by the isolated worker.

Back up the SQLite file if plan and audit history matter. Compiled caches can be
regenerated, but should only be reused with compatible models, runtimes and
drivers.

## Health and shutdown

- Use `/health` for a bounded HTTP process check.
- Use `AdaptiveExecutionEngine.readiness()` for local stored-state readiness.
- Use `direct_status()` for the isolated native worker.
- Close `KernelLoomModel` and `AdaptiveExecutionEngine` during graceful shutdown.

The FastAPI application factory closes all registered models through its
lifespan handler.

## Build a package locally

```bash
python -m pip install -e ".[dev,langchain,server,rag]"
python -m pytest
python -m build
python -m twine check dist/*
```

The expected artifacts are a universal wheel and a source distribution.

## Automatic PyPI publishing

The repository workflow at `.github/workflows/publish.yml` runs on pushes to
`main` and can also be started manually. A publish still depends on the GitHub
environment, the PyPI token, successful verification, and PyPI.

Configure a GitHub repository secret named `PYPI_API_TOKEN` containing a PyPI
project or account token. The workflow:

1. checks out the pushed commit;
2. runs the full test suite on Python 3.11, 3.12, and 3.13;
3. installs the server, LangChain, FastEmbed, and FAISS extras on the package
   build runner and runs the tests again;
4. checks that all source version declarations match and that PyPI does not
   already contain the release version;
5. builds the wheel and source distribution;
6. validates both artifacts with Twine;
7. publishes the artifacts to PyPI after the verification and build jobs pass.

PyPI filenames are immutable. Before a release, choose a new normal version
such as `0.4.1`; do not reuse an existing version. Update `project.version` in
`pyproject.toml`,
`kernelloom._SOURCE_VERSION`, and `openagent_engine.__version__` together.
When that commit reaches `main`, the workflow checks the exact version against
PyPI before it runs the build. If the version is already published, the
workflow stops and the next release must bump the version first.

## Forks and pull requests

The publishing workflow runs only for pushes to `main`, not for pull requests.
Fork repository secrets are not provided to upstream pull-request workflows.
Disable or rename the publishing workflow on a fork until its own PyPI token and
package name are configured.

## Release checklist

Before merging or pushing to `main`:

1. Run the local tests.
2. Build and validate both distributions.
3. Bump the version to an unused `major.minor.patch` release and verify every
   source declaration matches it.
4. Review README rendering for both GitHub and PyPI.
5. Confirm `PYPI_API_TOKEN` is current and scoped appropriately.
6. Push the version-bump commit to `main` and check the GitHub Actions run.
7. Install the published wheel in a clean environment and run an import smoke test.
