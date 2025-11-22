# RPP Engine Documentation

## Overview
RPPEngine supports hybrid template loading: local templates in `/templates/` and remote templates from a GitHub repo.

## Usage
- List templates: send `rpp_list_templates` via WebSocket.
- Generate: send `rpp_generate` with template name and context.

## Template formats
Supported formats: `.html`, `.md`, `.jinja`, `.j2`, `.txt`. Use Jinja2 placeholders for dynamic fields.

Example Jinja snippet in template:
```
## RPP - {{ subject }} Grade {{ grade }}

Tujuan pembelajaran:
- {{ learning_goal }}
```

## Export
- Markdown (default)
- PDF (requires wkhtmltopdf + pdfkit)

