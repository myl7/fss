#!/usr/bin/env python3
r"""Doxygen input filter that converts $...$ math syntax to \f$...\f$ syntax.

Reads a source file from the path given as argv[1] (Doxygen INPUT_FILTER
protocol) and writes the converted output to stdout.

For source files (.c, .cpp, .cu, .cuh, .h, etc.), only converts dollar-sign
math within comment blocks:
  - Line comments: /// or //!
  - Block comments: /** ... */ or /* ... */

For Markdown files (.md), converts math on all lines since the entire file
is documentation.

Conversions:
  $$...$$ → \f[...\f]   (display math)
  $...$  → \f$...\f$   (inline math)

Already-existing \f$ sequences and escaped \$ are preserved.
"""

import os
import re
import sys


def convert_math(text: str) -> str:
    """Convert $...$ and $$...$$ to Doxygen math commands in a comment string."""
    # Preserve existing \f$ and \f[ sequences by temporarily replacing them
    text = text.replace("\\f$", "\x00FDOLLAR\x00")
    text = text.replace("\\f[", "\x00FLBRACKET\x00")
    text = text.replace("\\f]", "\x00FRBRACKET\x00")

    # Preserve escaped \$
    text = text.replace("\\$", "\x00ESCAPED\x00")

    # Convert $$...$$ → \f[...\f] (display math, do this first)
    text = re.sub(r"\$\$(.+?)\$\$", r"\\f[\1\\f]", text)

    # Convert $...$ → \f$...\f$ (inline math)
    text = re.sub(r"\$(.+?)\$", r"\\f$\1\\f$", text)

    # Restore preserved sequences
    text = text.replace("\x00FDOLLAR\x00", "\\f$")
    text = text.replace("\x00FLBRACKET\x00", "\\f[")
    text = text.replace("\x00FRBRACKET\x00", "\\f]")
    text = text.replace("\x00ESCAPED\x00", "\\$")

    return text


def process(source: str, filepath: str) -> str:
    """Process a source file, converting math only inside comments.

    For Markdown files, all lines are treated as documentation.
    For source files, only comment blocks are processed.
    """
    _, ext = os.path.splitext(filepath)
    if ext.lower() in (".md", ".markdown"):
        return convert_math(source)

    in_block_comment = False
    result = []

    for line in source.splitlines(keepends=True):
        if in_block_comment:
            # Check if block comment ends on this line
            end_idx = line.find("*/")
            if end_idx != -1:
                # Convert the comment portion (up to and including */)
                comment_part = line[: end_idx + 2]
                rest = line[end_idx + 2 :]
                line = convert_math(comment_part) + rest
                in_block_comment = False
            else:
                line = convert_math(line)
        else:
            # Check for line comments: /// or //!
            m = re.match(r"^(.*?)(///|//!)(.*)", line, re.DOTALL)
            if m:
                prefix, marker, comment = m.groups()
                line = prefix + marker + convert_math(comment)
            else:
                # Check for block comment start: /* or /**
                # There may be code before the comment on the same line
                m = re.search(r"/\*", line)
                if m:
                    start_idx = m.start()
                    before = line[:start_idx]
                    after = line[start_idx:]

                    # Check if block comment also ends on this line
                    end_idx = after.find("*/", 2)
                    if end_idx != -1:
                        # Single-line block comment
                        comment_part = after[: end_idx + 2]
                        rest = after[end_idx + 2 :]
                        line = before + convert_math(comment_part) + rest
                    else:
                        # Block comment continues to next line
                        line = before + convert_math(after)
                        in_block_comment = True

        result.append(line)

    return "".join(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: doxygen-math-filter.py <filename>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        source = f.read()

    sys.stdout.write(process(source, sys.argv[1]))
