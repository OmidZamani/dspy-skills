# Changelog

All notable changes to DSPy Skills will be documented in this file.

## [Unreleased]

## [2.0.0] - 2026-06-01

### Added
- Seven DSPy 3.2.1 skills: optimizer selection, BetterTogether, reasoning modules, embedding retrieval, adapters and multimodal I/O, MCP tool integration, and production deployment.
- Repository validator and GitHub Actions workflow.
- DSPy 3.3 prerelease notes.

### Changed
- Updated the stable DSPy baseline from 3.1.2 to 3.2.1.
- Expanded the collection from 15 to 22 skills.
- Reworked the README into a task-oriented index.

### Fixed
- Replaced the invalid Claude Code marketplace skill catalog with a valid marketplace manifest.
- Updated GEPA examples to use five-argument feedback metrics returning `dspy.Prediction(score=..., feedback=...)`.
- Removed the obsolete `BootstrapFinetune` experimental flag.
- Fixed invalid `PythonInterpreter({})` examples.
- Corrected the original release date from 2025-01-20 to 2026-01-20.

## [1.1.0] - 2026-02-21

### Added
- `dspy-optimize-anything` for GEPA optimization of arbitrary text artifacts.
- ReAct, debugging, output refinement, SIMBA, advanced composition, and custom module skills.
- Haystack progressive-disclosure references and examples.

## [1.0.0] - 2026-01-20

### Added
- Initial DSPy skill collection.

[Unreleased]: https://github.com/OmidZamani/dspy-skills/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/OmidZamani/dspy-skills/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/OmidZamani/dspy-skills/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/OmidZamani/dspy-skills/releases/tag/v1.0.0
