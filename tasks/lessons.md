# Lessons Learned

## Shell Compatibility
- **Problem:** Using `&&` in command line runs on Windows PowerShell fails with parser errors (e.g. `The token '&&' is not a valid statement separator in this version.`).
- **Remedy:** Use `;` as the statement separator instead of `&&` in Windows PowerShell.

## LaTeX Formatting and Special Characters
- **Problem:** LaTeX requires special escaping for characters like `&` (`\&`), `%` (`\%`), `_` (`\_`), etc.
- **Remedy:** Always verify the correctness of escaping when adding or editing resume descriptions (e.g., `Backend \& Systems` or `Cloud \& DevOps`).
- **Remedy:** Ensure URLs do not contain unescaped special characters.

## README.md Rendering
- **Problem:** Complex HTML structures or bad badge URLs in markdown can cause rendering issues on GitHub.
- **Remedy:** Use clean, standard markdown syntax and test the layout/rendering. Avoid non-standard structures. Use standard Mermaid code blocks for diagrams.

## ATS / Digital PDF Parser Compatibility
- **Problem:** Many Applicant Tracking Systems (ATS) and PDF-to-text parsers fail to extract text accurately when using symbol/icon packages like `fontawesome5`. Icons get parsed as random unicode characters (e.g., `+`, `#`, `§`, `ï`), which clutters the data.
- **Problem:** Tight top margins (e.g. `top=0.5in`) can cause PDF parsers to clip the top-most line, dropping the first letter (e.g. "Ashwin Goyal" parsing as "shwin Goyal").
- **Remedy:** Avoid visual font icons in key contact lines. Rebuild headers with clean, standard text dividers (like `|` or `\bullet`).
- **Remedy:** Add `\vspace*{10pt}` right after `\begin{document}` to nudge the header down slightly, ensuring the parser registers the entire first line properly.

