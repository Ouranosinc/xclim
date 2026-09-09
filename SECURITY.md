# Security Policy

## Supported Versions

`xclim` is in rapid development and receives regular updates every four to six (4-6) weeks.
In the event of a security-related bug discovery soon after the release of an `xclim` version, the last supported version will receive a patch release.

## Security Requirements

`xclim` is an Open Source Python library and is designed to be run on both trusted platforms as well as on hosted compute platforms (such as [PAVICS](https://pavics.ouranos.ca)).
As such, the primary security focus of the `xclim` maintainers are to ensure the integrity of source code and packages as well as to respond to security vulnerabilities as they arise.

### Source Code

- All changes to the project are made through pull requests and are reviewed by at least one maintainer before being merged.
- While not a requirement for submitting changes to code, contributors are generally encouraged to cryptographically sign commits where practical.
- Releases are created from tagged revisions in the Git repository to ensure that released source code corresponds to a reviewed state of the project.

### Releases

- Python packages are published to PyPI using Trusted Publishing through GitHub Actions.
- Release artifacts are built automatically by the release workflow from the tagged source code.
- Release tags are cryptographically signed, and users may verify the authenticity of the source release using the published signing key and the project's release documentation.

### Dependencies

- Project dependencies are declared explicitly in `pyproject.toml` and `environment.yml`.
- Core library dependency updates are reviewed by maintainers before being merged.
- GitHub Actions and Python dependencies used by automated workflows are pinned to immutable commit hashes whenever possible.
- Automated dependency update tools and vulnerability scanning may be used to identify outdated, redundant, or vulnerable dependencies.
  - Automated tools such as Dependabot may perform non-significant ("minor"/"patch") unattended updates of GitHub Actions and other CI configurations while significant ("major") updates
    must be reviewed by maintainers.

## Reporting a Vulnerability

If you believe you have found a security vulnerability in `xclim`, we encourage you to let us know right away. We take all security vulnerabilities seriously and appreciate your efforts to responsibly disclose them.

Please follow these steps to report a security vulnerability:

1. **Email**: Email [github-support@ouranos.ca](mailto:github-support@ouranos.ca) with a detailed description of the vulnerability. If applicable, please include any steps or a proof-of-concept to help us understand and reproduce the issue.

1. **Encryption (Optional)**: If you are concerned about the sensitivity of the information you are sharing, you can use the PGP key found below to encrypt your communication.

1. **Response**: We will acknowledge your email within 48 hours and work with you to understand and confirm the vulnerability.

1. **Fix and Disclosure**: Once the vulnerability is confirmed, we will work to address it promptly. We appreciate your patience as we investigate and implement a fix. Once resolved, we will coordinate the disclosure and provide credit to the reporter unless they prefer to remain anonymous.

## PGP Encryption Key

You can use the following PGP key to encrypt your communications with us:

```
-----BEGIN PGP PUBLIC KEY BLOCK-----

mDMEZamQrhYJKwYBBAHaRw8BAQdA+saPvmvr1MYe1nQy3n3QDcRE9T7UzTJ1XH31
EI4Zb6u0Mk91cmFub3MgR2l0SHViIFN1cHBvcnQgPGdpdGh1Yi1zdXBwb3J0QG91
cmFub3MuY2E+iJkEExYKAEEWIQSeAu+Cbjupx79jy9VeVFD6o5TVcwUCZamQrgIb
AwUJCWYBgAULCQgHAgIiAgYVCgkICwIEFgIDAQIeBwIXgAAKCRBeVFD6o5TVc4ho
AQDXjDkx0b3A7yl6PQ4hBJ2uYzw0UWbml7mUwVdhMmdZkQD/VJZQNWrCQeOtYEM8
icZJYwR/OsKFOWqlDytusGGtjwa4OARlqZCuEgorBgEEAZdVAQUBAQdAa41Zabjz
P9O+p6tI69Cnft6U5om3+qCcMo8amTqauH0DAQgHiH4EGBYKACYWIQSeAu+Cbjup
x79jy9VeVFD6o5TVcwUCZamQrgIbDAUJCWYBgAAKCRBeVFD6o5TVcwmaAQClDxW6
2gir7lhRXAcO+vmRImpGd29TrkcQVh+ak7VlwQEA706d7Kusiorlf/h8pLSoNMmS
kuLGmHpUJ8NVGppU+wo=
=wuxr
-----END PGP PUBLIC KEY BLOCK-----
```
