from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import json
import sys

src = Path(sys.argv[1])
out = Path(sys.argv[2])
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

with ZipFile(src) as zf:
    root = etree.fromstring(zf.read("word/document.xml"))

rows = []
for idx, p in enumerate(root.xpath("//w:body//w:p", namespaces=ns)):
    text = "".join(p.xpath(".//w:t/text()", namespaces=ns))
    pstyle = p.find("w:pPr/w:pStyle", namespaces=ns)
    style = pstyle.get(f"{{{ns['w']}}}val") if pstyle is not None else ""
    has_drawing = bool(p.xpath(".//w:drawing|.//w:pict", namespaces=ns))
    rows.append({"index": idx, "style": style, "has_drawing": has_drawing, "text": text})

out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
joined = "\n".join(r["text"] for r in rows)
print(json.dumps({
    "paragraphs": len(rows),
    "nonempty": sum(bool(r["text"].strip()) for r in rows),
    "words": len(joined.split()),
    "characters": len(joined),
    "drawing_paragraphs": [r["index"] for r in rows if r["has_drawing"]],
}, ensure_ascii=False, indent=2))
