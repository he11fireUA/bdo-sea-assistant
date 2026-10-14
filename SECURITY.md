# Security Policy

## Reporting a vulnerability

Do not open a public issue for vulnerabilities, leaked credentials, private captures, or reports containing player identities.

Use GitHub private vulnerability reporting for this repository. If that is unavailable, contact the maintainer through the GitHub account `@he11fireUA` without posting sensitive evidence publicly.

Include:

- affected version or commit;
- reproduction steps;
- practical impact;
- the smallest redacted log or fixture needed to reproduce the issue.

Do not upload raw game recordings, configuration files, tokens, local paths, or unrelated personal data.

## Scope

Security-sensitive areas include update verification, release signing, configuration handling, local file paths, Windows process launching, capture boundaries, and untrusted fixture parsing.

The project does not read game memory, inject code, hook DirectX, send game input, or modify game files. Reports claiming otherwise should include a reproducible code path.
