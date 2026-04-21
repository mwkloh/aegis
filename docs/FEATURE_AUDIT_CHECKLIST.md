# AEGIS — Feature Security Audit Checklist

Use this checklist before adding any new feature.

## Prompt Safety
- [ ] Does user input remain DATA only?
- [ ] Are system prompts file-based?

## Tool Safety
- [ ] Are tools scoped by skill?
- [ ] Can language bypass tool permissions?

## Memory Safety
- [ ] Are memory writes proposal-only?
- [ ] Is identity or preference immutable at runtime?

## Improvement Safety
- [ ] Are changes out-of-band?
- [ ] Is human approval required?

## Coding Harness Safety
- [ ] Are outputs diff-only?
- [ ] Is deployment manual?

## If any answer is NO
Stop and redesign before implementation.
