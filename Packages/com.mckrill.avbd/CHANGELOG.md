# Changelog

All notable changes to this package are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- Package skeleton: manifest, assembly definitions, folder layout and documentation stubs.
- Smoke compute kernel and tests that prove assemblies compile, the shader is found and a dispatch works.
- CPU reference solver math layer (`McKrill.Avbd.Reference`): `Vec6`, `Mat6` with an LDL^T solver, and quaternion helpers (`AvbdQuat`), with Edit Mode tests.
