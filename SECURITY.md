# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.0.x | yes |

Only the latest release gets fixes.

## Reporting a vulnerability

Please do **not** open a public issue, pull request or discussion for a security problem.

Report it privately to the maintainers listed in [.github/CODEOWNERS](.github/CODEOWNERS), through GitHub (a direct message or the contact on their profile). Include:

- what is affected (file, command or setting) and the version (`dsd-tuner --version`);
- how to reproduce it, and what an attacker could do with it;
- any fix you suggest.

We aim to acknowledge a report within 3 working days and to agree a fix and disclosure date with you.

## Scope

dsd-tuning is a tuner and a vLLM plugin. It reads the files and environment variables you give it (tables, step costs, acceptance, `VLLM_DSD_*`) and runs inside your vLLM server. Treat table folders like code: only serve tables from sources you trust. Security problems in vLLM itself belong to the [vLLM project](https://github.com/vllm-project/vllm/security).
