---
name: commit-msg
description: Use when writing commit messages.
---

```
[optional scope: "<scope>: "]<description>

[optional body]

[optional trailer(s)]
```

- `<scope>` — the subsystem, area, or module that the commit touches
- `<description>` — a short description of the changes made
- `[optional body]` — detailed information about the changes
- `[optional trailer(s)]` — additional metadata about the commit

Use the sentence case for the description, e.g., "Add license".
Omit the period "." when the description has only one sentence.
Make it quickly understandable to contributors and as short as possible.

Use singular words for scopes.
The predefined scopes: ci, dep, doc, bench, build, example
Must not use the scopes: feat, fix, change
Add the scope only when the predefined ones or the ones explicitly defined by the user with some files or prompts are used.
Scopes used in previous commit msgs are not the ones explicitly defined.

Do not add the body when the code diff can effectively express the commit changes.
