"""Unified document representation shared by every converter.

All parsers (MinerU, MarkItDown, plain text) produce the same JSON document:

{
  "schema": "rag.document",
  "schema_version": "1.0",
  "source": {"filename", "ext", "sha256", "bytes", "converter", "converter_version", "parsed_at"},
  "metadata": {"page_count", "block_count", "char_count", "image_count"},
  "pages": [{"page_idx": 0, "blocks": [{"type", "text", "level"?, "bbox"?, "image"?}]}],
  "images": [{"name", "object_key"? , "data_uri"?}],
  "markdown": "..."
}

`markdown` is the canonical flattened text used for chunking and for the
`<name>.md` download; `pages`/`blocks` preserve layout so downstream code can
re-read structure (tables, headings, bboxes) without re-parsing the original.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone

SCHEMA_ID = "rag.document"
SCHEMA_VERSION = "1.0"

BLOCK_TYPES = {"text", "title", "table", "image", "equation", "list", "code", "page_number"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _block(block_type: str, text: str = "", *, level: int | None = None,
           bbox: list | None = None, image: str | None = None,
           **extra) -> dict:
    if block_type not in BLOCK_TYPES:
        block_type = "text"
    item: dict = {"type": block_type}
    if text:
        item["text"] = text
    if level is not None:
        item["level"] = level
    if bbox:
        item["bbox"] = bbox
    if image:
        item["image"] = image
    item.update({k: v for k, v in extra.items() if v not in (None, "", [], {})})
    return item


def build_document(filename: str, raw: bytes | None, pages: list[dict],
                   markdown: str, *, converter: str, converter_version: str = "",
                   images: list[dict] | None = None,
                   metadata: dict | None = None) -> dict:
    """Assemble the unified document and recompute summary metadata."""
    markdown = (markdown or "").strip()
    blocks = [b for page in pages for b in page.get("blocks", [])]
    meta = {
        "page_count": len(pages),
        "block_count": len(blocks),
        "char_count": len(markdown),
        "image_count": len(images or []),
    }
    meta.update(metadata or {})
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "filename": filename,
            "ext": os.path.splitext(filename)[1].lower(),
            "sha256": sha256_bytes(raw) if raw is not None else "",
            "bytes": len(raw) if raw is not None else 0,
            "converter": converter,
            "converter_version": converter_version,
            "parsed_at": now_iso(),
        },
        "metadata": meta,
        "pages": pages,
        "images": images or [],
        "markdown": markdown,
    }


# --------------------------------------------------------------- markdown
def blocks_to_markdown(pages: list[dict]) -> str:
    """Render normalized blocks back to Markdown (used as a fallback)."""
    out: list[str] = []
    for page in pages:
        for block in page.get("blocks", []):
            kind = block.get("type")
            text = (block.get("text") or "").strip()
            if kind == "title" and text:
                level = min(max(int(block.get("level") or 1), 1), 6)
                out.append("#" * level + " " + text)
            elif kind == "table" and text:
                out.append(text)
            elif kind == "image":
                name = block.get("image") or ""
                caption = text
                if name:
                    out.append(f"![{caption}]({name})")
                elif caption:
                    out.append(caption)
            elif kind == "equation" and text:
                out.append(text if text.startswith("$$") else f"$${text}$$")
            elif kind == "list" and text:
                out.append(text)
            elif kind == "code" and text:
                out.append("```\n" + text + "\n```")
            elif kind == "page_number":
                continue
            elif text:
                out.append(text)
    return "\n\n".join(out).strip()


def document_text(document: dict) -> str:
    """Flatten a unified document into text suitable for chunking."""
    markdown = (document.get("markdown") or "").strip()
    if markdown:
        return markdown
    return blocks_to_markdown(document.get("pages") or [])


# --------------------------------------------------------- local converters
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".log"}


def _markdown_source_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    for chunk in re.split(r"\n{2,}", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", chunk, re.S)
        if heading and "\n" not in heading.group(2).strip():
            blocks.append(_block("title", heading.group(2).strip(), level=len(heading.group(1))))
        else:
            blocks.append(_block("text", chunk))
    return blocks


def parse_text_file(path: str, filename: str, raw: bytes | None = None) -> dict:
    """Plain text/Markdown/CSV: decode locally, no external service."""
    ext = os.path.splitext(filename)[1].lower()
    if raw is not None:
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to decode text file")
    else:
        text = ""
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                with open(path, "r", encoding=encoding) as handle:
                    text = handle.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to decode text file")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if ext in {".md", ".markdown"}:
        blocks = _markdown_source_blocks(text)
        markdown = text.strip()
    else:
        blocks = [_block("text", text.strip())] if text.strip() else []
        markdown = text.strip()
    pages = [{"page_idx": 0, "blocks": blocks}] if blocks else []
    return build_document(filename, raw if raw is not None else text.encode("utf-8"),
                          pages, markdown, converter="text", converter_version="local")


def parse_with_markitdown(path: str, filename: str, raw: bytes | None = None) -> dict:
    """Office/HTML and any other MarkItDown-supported format."""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError("MarkItDown is not installed in the rag environment") from exc
    result = MarkItDown().convert(path)
    markdown = (result.text_content or "").strip()
    blocks = _markdown_source_blocks(markdown)
    pages = [{"page_idx": 0, "blocks": blocks}] if blocks else []
    return build_document(filename, raw, pages, markdown,
                          converter="markitdown", converter_version="markitdown")


# --------------------------------------------------------------- MinerU
def _strip_markdown_prefix(text: str) -> str:
    return re.sub(r"^#{1,6}\s*", "", text.strip())


def normalize_content_list(content: list[dict], markdown: str | None = None,
                           images: dict[str, str] | None = None) -> tuple[list[dict], str]:
    """Convert MinerU `content_list.json` into unified pages/blocks.

    MinerU 2.x item types: text, title, image, table, equation, list, code,
    page_number. Fields outside the known set are preserved in `extra`.
    """
    pages: dict[int, dict] = {}
    images = images or {}
    rendered: list[str] = []

    for item in content or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "text").strip().lower()
        page_idx = int(item.get("page_idx") or 0)
        bbox = item.get("bbox") if isinstance(item.get("bbox"), (list, tuple)) else None
        bbox = list(bbox) if bbox else None
        text = ""
        image_name = None
        _extra: dict = {}

        # MinerU 3.x emits most headings as `type: "text"` with `text_level` set;
        # only older versions used a dedicated "title" type.
        text_level = item.get("text_level")
        if kind == "text" and text_level:
            kind = "title"

        if kind == "title":
            text = _strip_markdown_prefix(item.get("text") or "")
            level = int(text_level or item.get("level") or 1)
            block = _block("title", text, level=level, bbox=bbox, **_extra)
            rendered.append("#" * min(max(level, 1), 6) + " " + text if text else "")
        elif kind == "table":
            text = (item.get("table_body") or "").strip()
            caption = " ".join(item.get("table_caption") or [])
            footnote = " ".join(item.get("table_footnote") or [])
            image_name = item.get("img_path")
            if caption:
                _extra["caption"] = caption
            if footnote:
                _extra["footnote"] = footnote
            block = _block("table", text, bbox=bbox, image=image_name, **_extra)
            if caption:
                rendered.append(f"**{caption}**")
            rendered.append(text or (f"![table]({image_name})" if image_name else ""))
            if footnote:
                rendered.append(footnote)
        elif kind == "image":
            image_name = item.get("img_path")
            caption = " ".join(item.get("img_caption") or [])
            footnote = " ".join(item.get("img_footnote") or [])
            if caption:
                _extra["caption"] = caption
            if footnote:
                _extra["footnote"] = footnote
            block = _block("image", caption, bbox=bbox, image=image_name, **_extra)
            if image_name:
                rendered.append(f"![{caption}]({image_name})")
            if caption:
                rendered.append(caption)
            if footnote:
                rendered.append(footnote)
        elif kind == "equation":
            text = (item.get("text") or "").strip()
            block = _block("equation", text, bbox=bbox,
                           text_format=item.get("text_format"))
            if text:
                rendered.append(text if text.startswith("$$") else f"$${text}$$")
        elif kind == "list":
            items = [str(x).strip() for x in (item.get("list_items") or []) if str(x).strip()]
            text = "\n".join(f"- {entry}" for entry in items)
            block = _block("list", text, bbox=bbox)
            if text:
                rendered.append(text)
        elif kind == "code":
            text = (item.get("code_body") or "").strip()
            caption = " ".join(item.get("code_caption") or [])
            if caption:
                _extra["caption"] = caption
            block = _block("code", text, bbox=bbox, **_extra)
            if caption:
                rendered.append(f"**{caption}**")
            if text:
                rendered.append("```\n" + text + "\n```")
        elif kind == "page_number":
            block = _block("page_number", item.get("text") or "", bbox=bbox)
        else:
            text = (item.get("text") or "").strip()
            block = _block("text", text, bbox=bbox)
            if text:
                rendered.append(text)

        if block.get("image") and block["image"] not in images:
            # Content list may reference an image the service did not return.
            block.pop("image", None)
        pages.setdefault(page_idx, {"page_idx": page_idx, "blocks": []})["blocks"].append(block)

    ordered = [pages[key] for key in sorted(pages)]
    markdown_out = (markdown or "").strip() or "\n\n".join(x for x in rendered if x).strip()
    return ordered, markdown_out


def build_mineru_document(filename: str, raw: bytes | None, content_list: list[dict],
                          markdown: str, *, backend: str = "", version: str = "",
                          images: dict[str, str] | None = None) -> dict:
    pages, markdown_out = normalize_content_list(content_list, markdown, images)
    image_entries = [{"name": name, "data_uri": uri} for name, uri in (images or {}).items()]
    return build_document(
        filename, raw, pages, markdown_out,
        converter="mineru",
        converter_version=version or backend or "mineru",
        images=image_entries,
        metadata={"backend": backend, "mineru_version": version},
    )


def dump_json(document: dict) -> bytes:
    return json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")


def load_json(payload: bytes | str) -> dict:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)
