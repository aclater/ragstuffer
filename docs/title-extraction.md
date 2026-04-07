# Title extraction

Titles are extracted per source type and stored alongside chunk metadata in Postgres, then surfaced by ragpipe in `rag_metadata.cited_chunks[].title`.

Title extraction enables ragpipe to surface document titles in citations without storing full text in Qdrant.

## Title source by format

| Source type | Title extraction |
|-------------|------------------|
| Google Drive (PDF) | PDF metadata `Title` field, or filename without extension |
| Google Drive (DOCX/PPTX) | Document title from Office metadata, or filename |
| Google Drive (XLSX) | Sheet name or filename |
| Google Drive (other) | Filename |
| git repos | First Markdown heading (`# Title`) in file, or filename |
| Web URLs | `<title>` tag content, or URL path |
| Local files | Filename |

## How titles are used

Titles are stored in the `documents` table in Postgres alongside chunk metadata. When ragpipe retrieves chunks for a query, it includes the title in `rag_metadata.cited_chunks[].title`, allowing responses to cite documents by their proper titles rather than internal IDs.
