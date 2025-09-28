# Entity Query Language Guidelines

## Testing
- If you need to run tests, execute them with pytest
- Reuse existing fixtures
- Always use a test-driven development approach

## Code Style
- Do not use abbreviations
- Create classes instead of using too many primitives
- Minimize duplication of code
- Use the semantic_world/doc/styleguide.md for naming transformation variables
- Do not wrap attribute access in try-except blocks
- Always access attributes via dot notation, never via getattr
- Use existing packages whenever possible
- Reduce nesting, reduce complexity
- Use short but descriptive names

## Design Principles
- Always apply the SOLID principles of object-oriented programming 
- If you have to create new view classes, stick to the design of existing views found in semantic_world.views.views
- Create meaningful custom exceptions
- Eliminate YAGNI smells
- Make interfaces hard to misuse

## Documentation
- Write docstrings that explain what the function does and not how it does it
- Do not create type information for docstring

## Misc
- If you find a package that could be replaced by a more powerful one, let us know