import html
import re


def html_to_plain_text(text):
    if not isinstance(text, str):
        return text
    if "<" not in text or ">" not in text:
        return html.unescape(text)
    cleaned = text
    cleaned = re.sub(r"(?i)</tr\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</p\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)</div\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)</td\s*>\s*<td[^>]*>", " ", cleaned)
    cleaned = re.sub(r"(?i)<[^>]+>", "", cleaned)
    cleaned = html.unescape(cleaned)
    return cleaned.strip()


def normalize_text_lines(text):
    if not isinstance(text, str):
        return []
    text = html_to_plain_text(text)
    normalized_lines = []
    paragraph_lines = []

    def flush_paragraph():
        if not paragraph_lines:
            return
        normalized_lines.append(" ".join(paragraph_lines))
        paragraph_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            paragraph_lines.append(line)
        else:
            flush_paragraph()
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")

    flush_paragraph()

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()
    return normalized_lines


def extract_layout_block_lines(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            item_lines = extract_layout_block_lines(item)
            if item_lines:
                if lines and lines[-1] != "":
                    lines.append("")
                lines.extend(item_lines)
        return lines

    if isinstance(value, dict):
        blocks = value.get("parsing_res_list")
        if isinstance(blocks, list):
            return extract_lines_from_layout_blocks(blocks)
        return []

    blocks = getattr(value, "parsing_res_list", None)
    if isinstance(blocks, list):
        return extract_lines_from_layout_blocks(blocks)

    return []


def extract_lines_from_layout_blocks(blocks):
    lines = []
    for block in blocks:
        content = None
        if isinstance(block, dict):
            content = block.get("block_content") or block.get("content")
        else:
            content = getattr(block, "content", None)
        block_lines = normalize_text_lines(content)
        if not block_lines:
            continue
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(block_lines)

    while lines and lines[-1] == "":
        lines.pop()
    return lines


def extract_text_lines(value):
    lines = []

    if value is None:
        return lines

    if isinstance(value, str):
        return normalize_text_lines(value)

    if isinstance(value, (list, tuple)):
        for item in value:
            lines.extend(extract_text_lines(item))
        return lines

    if isinstance(value, dict):
        for key in (
            "markdown",
            "text",
            "content",
            "rec_text",
            "rec_texts",
            "parsing_res_list",
            "layout_parsing_result",
        ):
            if key in value:
                extracted = extract_text_lines(value[key])
                if extracted:
                    return extracted
        return []

    for attr in ("markdown", "text", "content", "rec_text", "rec_texts", "json", "res"):
        if hasattr(value, attr):
            attr_value = getattr(value, attr)
            if callable(attr_value):
                continue
            lines.extend(extract_text_lines(attr_value))

    return lines
