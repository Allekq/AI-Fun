These are language-agnostic coding principles to keep code readable, maintainable, and robust.

## Constants & Configuration
- Avoid hardcoded values; surface any configurable value as a constant or config entry.
- Use clear naming for constants (e.g., UPPER_SNAKE_CASE).
- Centralize configuration (dedicated config/constants module or file).
- Use enums or equivalent for fixed-choice options and store runtime values in the enum.

## Data Structures & Types
- Use structured types (dataclasses/interfaces) for multi-field data.
- Prefer simple primitives for single-field values.
- Keep domain models grouped logically.
- Maintain explicit types where supported; prefer compile-time/type-checker enforcement.

## Naming & Style
- Files: snake_case
- Classes: PascalCase
- Functions: snake_case
- Names should describe intent, not implementation.

## Separation of Concerns
- One responsibility per module/file.
- Separate public API (entry points) from implementation details.
- Utilities and helpers should live in dedicated modules.

## Error Handling & Validation
- Validate inputs early and fail fast with clear errors.
- Use explicit error types/exceptions rather than generic ones.
- Don’t swallow errors silently; log or propagate appropriately.

## Type Safety
- No `# type: ignore` or `noqa` comments
- Use TypedDict for dict structures with known shapes
- Code must compile without type errors

## Code Organization
- DRY (dont repete yourself) rules must be respected
- Utility functions in dedicated files 
- One clear responsibility per file
- Clean separation: core logic → utilities → entry points

## Imports
- Use explicit relative imports within packages
- Re-export public API in `__init__.py` for clean imports
