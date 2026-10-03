from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


SRC = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_paginacao_corrigida.docx")
DST = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_paginacao_sequencial.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
Q = lambda name: f"{{{W}}}{name}"


with TemporaryDirectory() as temp_dir:
    temp = Path(temp_dir)
    with ZipFile(SRC) as zin:
        zin.extractall(temp)

    document_path = temp / "word/document.xml"
    tree = etree.parse(str(document_path))
    root = tree.getroot()

    sections = root.xpath("//w:sectPr", namespaces=NS)
    if len(sections) != 4:
        raise RuntimeError(f"Esperadas 4 seções, encontradas {len(sections)}")

    third_pg_num = sections[2].find(Q("pgNumType"))
    if third_pg_num is None:
        third_pg_num = etree.SubElement(sections[2], Q("pgNumType"))
    third_pg_num.set(Q("start"), "7")
    third_pg_num.set(Q("fmt"), "decimal")

    paragraphs = root.xpath("//w:body/w:p", namespaces=NS)
    in_toc = False
    updated = []
    for index, paragraph in enumerate(paragraphs[:100]):
        text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()
        if text.startswith("1 INTRODUÇÃO"):
            in_toc = True
        if not in_toc:
            continue

        numeric_nodes = [
            node
            for node in paragraph.xpath(".//w:t", namespaces=NS)
            if (node.text or "").strip().isdigit()
        ]
        if numeric_nodes:
            old = int(numeric_nodes[-1].text.strip())
            new = old - 4
            numeric_nodes[-1].text = str(new)
            updated.append((index, text, old, new))

        if text.startswith("APÊNDICE A"):
            break

    if len(updated) != 27:
        raise RuntimeError(f"Esperadas 27 entradas do sumário, atualizadas {len(updated)}")

    document_path.write_bytes(
        etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
    )

    settings_path = temp / "word/settings.xml"
    settings = etree.parse(str(settings_path))
    settings_root = settings.getroot()
    update = settings_root.find(Q("updateFields"))
    if update is None:
        update = etree.SubElement(settings_root, Q("updateFields"))
    update.set(Q("val"), "true")
    settings_path.write_bytes(
        etree.tostring(settings, xml_declaration=True, encoding="UTF-8", standalone=True)
    )

    with ZipFile(DST, "w", ZIP_DEFLATED) as zout:
        for path in sorted(temp.rglob("*")):
            if path.is_file():
                zout.write(path, path.relative_to(temp).as_posix())

print(f"Criado: {DST}")
print("Terceira seção reiniciada em 7; quarta seção permanece contínua.")
for index, text, old, new in updated:
    print(f"TOC p{index}: {old} -> {new} | {text[:70]}")
