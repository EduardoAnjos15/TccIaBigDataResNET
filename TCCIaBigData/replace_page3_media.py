from pathlib import Path
from zipfile import ZipFile

from lxml import etree


SOURCE_DOCX = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_ficha_catalografica.docx")
REPLACEMENT_PNG = Path(r"D:\TCCIaBigData\tmp\file_render\file.png")
OUTPUT_DOCX = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_pagina3_file.docx")

NS = {
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def locate_page3_image(archive: ZipFile) -> str:
    document = etree.fromstring(archive.read("word/document.xml"))
    relationships = etree.fromstring(archive.read("word/_rels/document.xml.rels"))

    anchors = document.xpath('.//wp:anchor[@behindDoc="1"]', namespaces=NS)
    if len(anchors) != 1:
        raise RuntimeError(f"Expected one full-page background image, found {len(anchors)}")

    relationship_ids = anchors[0].xpath(".//a:blip/@r:embed", namespaces=NS)
    if len(relationship_ids) != 1:
        raise RuntimeError("Could not identify the page 3 image relationship")

    targets = relationships.xpath(
        "./pr:Relationship[@Id=$rid]/@Target",
        namespaces=NS,
        rid=relationship_ids[0],
    )
    if len(targets) != 1:
        raise RuntimeError("Could not resolve the page 3 image target")

    return "word/" + targets[0].lstrip("/")


def main():
    replacement = REPLACEMENT_PNG.read_bytes()

    with ZipFile(SOURCE_DOCX, "r") as source:
        target = locate_page3_image(source)
        if target not in source.namelist():
            raise RuntimeError(f"Image part not found: {target}")

        with ZipFile(OUTPUT_DOCX, "w") as output:
            for item in source.infolist():
                data = replacement if item.filename == target else source.read(item.filename)
                output.writestr(item, data)

    print(OUTPUT_DOCX)
    print(f"Replaced only {target}")


if __name__ == "__main__":
    main()
