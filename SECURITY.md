# Security policy

## Supported versions

MedDeID is currently pre-1.0 software. Security fixes are applied to the latest
published minor release of this component. Older releases are unsupported unless
a maintainer explicitly states otherwise in a security advisory.

## Reporting a vulnerability

Use GitHub's private **Report a vulnerability** form in this repository's
Security tab. Do not open a public issue for vulnerabilities, credentials,
patient information, private clinical text, or infrastructure details.

Include the affected version, deployment mode, impact, and a minimal
non-sensitive reproduction. The maintainers will acknowledge a report within
seven calendar days and coordinate disclosure after a fix or mitigation exists.

## Clinical safety boundary

A successful de-identification run does not guarantee anonymity. Deployments
processing clinical information require representative local validation,
access controls, monitoring, incident response, and institutional privacy and
security approval. Never attach real clinical text to a vulnerability report.
