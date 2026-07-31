# Security

## Supported versions

Only the latest released version of TinyContext receives security fixes.

## Docker image hardening

Each release image published to Docker Hub goes through the following before
it is pushed:

- **Config scan** — the `Dockerfile` is scanned with
  [Trivy](https://github.com/aquasecurity/trivy) for misconfigurations
  (`CRITICAL`/`HIGH`), and the build fails if any are found.
- **Vulnerability scan** — the built image is scanned with Trivy for known
  CVEs (`CRITICAL`/`HIGH`, fixed-only), and the build fails if any are found.
- **Non-root runtime** — the container drops from root to an unprivileged
  `tinycontext` user via `gosu` after fixing ownership of the mounted `/data`
  (and `/config`, if present) volume; the process itself never runs as root.
- **Minimal dependencies** — `pip` and `setuptools` are uninstalled from the
  image after install, so the release image ships only what's needed to run
  TinyContext.
- **Signed images** — published images are signed keylessly with
  [Cosign](https://github.com/sigstore/cosign) via GitHub OIDC. Verify with:

  ```sh
  cosign verify \
    --certificate-identity-regexp "https://github.com/TinySuiteHQ/TinyContext/.github/workflows/docker-publish.yml@.*" \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    marcellm01/tinycontext:<tag>
  ```

See [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml)
for the exact steps.

## Reporting a vulnerability

Please report security issues privately via [GitHub Security Advisories](https://github.com/TinySuiteHQ/TinyContext/security/advisories/new)
rather than filing a public issue. Include reproduction steps and the
affected version. We'll acknowledge reports within a few days.
