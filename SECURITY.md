# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly. **Do not open a public GitHub issue.**

### How to Report

1. Email the maintainers with a description of the vulnerability.
2. Include steps to reproduce the issue if possible.
3. Provide any relevant logs, screenshots, or proof-of-concept code.

### What to Expect

- **Acknowledgment** within 48 hours of your report.
- **Initial assessment** within 5 business days, including severity classification.
- **Resolution timeline** communicated once the issue is confirmed. Critical vulnerabilities will be prioritized.
- **Credit** given to the reporter in the release notes (unless you prefer to remain anonymous).

### Scope

The following are in scope for security reports:

- Vulnerabilities in the `automl-model-training` source code
- Dependency vulnerabilities that directly affect this project
- Insecure default configurations
- Path traversal or arbitrary file write via CLI arguments

The following are out of scope:

- Vulnerabilities in AutoGluon or other upstream dependencies (report those to the respective projects)
- Issues that require physical access to the machine running the software
- Social engineering attacks

## Security Best Practices for Users

- Keep dependencies up to date: run `uv sync --upgrade` periodically
- Do not expose trained model directories or prediction outputs to untrusted users — they may contain sensitive data from your training set
- Review `--output-dir` paths to avoid writing artifacts to unintended locations
- Run training and prediction in isolated environments when working with sensitive data
