# Contributing to DSPy Skills

Thank you for your interest in contributing to the DSPy Skills collection!

## How to Contribute

### Reporting Issues

- Check existing issues first
- Provide clear description and reproduction steps
- Include DSPy version and environment details

### Suggesting New Skills

Open an issue with:
- Skill name and purpose
- Target DSPy functionality
- Example use cases
- Why it doesn't fit in existing skills

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Make your changes
4. Test that examples run correctly
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

### Skill Guidelines

**Required Elements:**
- YAML frontmatter with name, version, dspy-compatibility, description (trigger-based), allowed-tools
- Goal section
- When to Use section
- Related Skills section (if applicable)
- Inputs/Outputs tables
- Phase-by-phase workflow
- Production example with error handling
- Best Practices section (3-5 items)
- Limitations section

**Description Pattern:**
```yaml
description: This skill should be used when the user asks to "[trigger]", "[trigger2]", mentions "[concept]", or [scenario].
```

**Code Quality:**
- Include error handling
- Add type hints
- Use realistic examples
- Follow DSPy best practices

**Skill Word Count:**
- Target: 400-650 words
- Maximum: 800 words
- Use progressive disclosure (references/, examples/) for longer content

## Questions?

Open an issue labeled "question" - we're happy to help!
