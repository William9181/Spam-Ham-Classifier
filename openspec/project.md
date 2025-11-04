# Project Context

## Purpose
This project implements an OpenSpec-driven development workflow for managing project specifications, changes, and documentation. It provides a structured approach to planning and implementing changes while maintaining clear documentation of system capabilities.

## Tech Stack
- Markdown for documentation and specifications
- Git for version control
- OpenSpec tooling for managing specifications
- GitHub for collaboration and code hosting

## Project Conventions

### Code Style
- Markdown files follow standard GitHub Flavored Markdown
- Spec files must follow OpenSpec formatting rules
- Change IDs use kebab-case with verb-led prefixes (add-, update-, remove-, refactor-)
- Requirements use SHALL/MUST for normative statements
- Scenarios use #### header level with WHEN/THEN format

### Architecture Patterns
- Specs are organized by capability in openspec/specs/
- Changes are tracked in openspec/changes/
- Each change follows proposal/tasks/design/specs structure
- Completed changes are archived in changes/archive/

### Testing Strategy
- Each requirement must have at least one scenario
- Scenarios define acceptance criteria
- Changes must pass openspec validate --strict before approval
- Implementation must satisfy all scenarios

### Git Workflow
- Main branch contains current production state
- Change proposals use feature branches
- PRs require passing OpenSpec validation
- Archive changes after deployment

## Domain Context
- OpenSpec-driven development workflow
- Specification-first approach to changes
- Clear separation between proposals and implemented specs

## Important Constraints
- All changes must have corresponding OpenSpec proposals
- Breaking changes must be clearly marked
- Each requirement needs at least one scenario
- Specs must be kept in sync with implementation

## External Dependencies
- Git version control
- OpenSpec CLI tools
- Markdown renderer
- GitHub (recommended)
