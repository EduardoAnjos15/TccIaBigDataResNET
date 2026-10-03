from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


SRC = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT.docx")
DST = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_paginacao_corrigida.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
qn = lambda name: f"{{{W}}}{name}"


def make_page_field(cached_value: str):
    runs = []

    begin = etree.Element(qn("r"))
    etree.SubElement(begin, qn("rPr"))
    fld_begin = etree.SubElement(begin, qn("fldChar"))
    fld_begin.set(qn("fldCharType"), "begin")
    runs.append(begin)

    instr_run = etree.Element(qn("r"))
    etree.SubElement(instr_run, qn("rPr"))
    instr = etree.SubElement(instr_run, qn("instrText"))
    instr.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    instr.text = " PAGE "
    runs.append(instr_run)

    separate = etree.Element(qn("r"))
    etree.SubElement(separate, qn("rPr"))
    fld_sep = etree.SubElement(separate, qn("fldChar"))
    fld_sep.set(qn("fldCharType"), "separate")
    runs.append(separate)

    value_run = etree.Element(qn("r"))
    etree.SubElement(value_run, qn("rPr"))
    value = etree.SubElement(value_run, qn("t"))
    value.text = cached_value
    runs.append(value_run)

    end = etree.Element(qn("r"))
    etree.SubElement(end, qn("rPr"))
    fld_end = etree.SubElement(end, qn("fldChar"))
    fld_end.set(qn("fldCharType"), "end")
    runs.append(end)
    return runs


footer_cached_values = {
    "word/footer6.xml": "26",  # even pages in section beginning at 11
    "word/footer7.xml": "25",  # odd pages in section beginning at 11
    "word/footer9.xml": "28",  # even pages in final section
    "word/footer10.xml": "27", # odd pages in final section
}

toc_updates = {
    "5 CONCLUSÃO": "24",
    "REFERÊNCIAS": "25",
    "APÊNDICE A": "27",
}


with TemporaryDirectory() as temp_dir:
    temp = Path(temp_dir)
    with ZipFile(SRC) as zin:
        zin.extractall(temp)

    for rel_path, cached in footer_cached_values.items():
        path = temp / rel_path
        root = etree.parse(str(path)).getroot()
        paragraph = root.find(qn("p"))
        if paragraph is None:
            raise RuntimeError(f"Rodapé sem parágrafo: {rel_path}")
        for child in list(paragraph):
            if child.tag != qn("pPr"):
                paragraph.remove(child)
        for run in make_page_field(cached):
            paragraph.append(run)
        path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True))

    document_path = temp / "word/document.xml"
    document = etree.parse(str(document_path))
    updated = {}
    paragraphs = document.xpath("//w:body/w:p", namespaces=NS)
    for index, paragraph in enumerate(paragraphs[:100]):
        text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()
        for prefix, new_page in toc_updates.items():
            if text.startswith(prefix):
                nodes = paragraph.xpath(".//w:t", namespaces=NS)
                numeric_nodes = [node for node in nodes if (node.text or "").strip().isdigit()]
                if not numeric_nodes:
                    raise RuntimeError(f"Número não encontrado no sumário: {text}")
                numeric_nodes[-1].text = new_page
                updated[prefix] = (index, new_page)
                break
    if set(updated) != set(toc_updates):
        raise RuntimeError(f"Entradas do sumário não atualizadas: {updated}")
    document_path.write_bytes(etree.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True))

    settings_path = temp / "word/settings.xml"
    settings = etree.parse(str(settings_path))
    root = settings.getroot()
    update = root.find(qn("updateFields"))
    if update is None:
        update = etree.SubElement(root, qn("updateFields"))
    update.set(qn("val"), "true")
    settings_path.write_bytes(etree.tostring(settings, xml_declaration=True, encoding="UTF-8", standalone=True))

    with ZipFile(DST, "w", ZIP_DEFLATED) as zout:
        for path in sorted(temp.rglob("*")):
            if path.is_file():
                zout.write(path, path.relative_to(temp).as_posix())

print(f"Criado: {DST}")
print(f"Rodapés convertidos para PAGE: {', '.join(footer_cached_values)}")
print(f"Sumário atualizado: {updated}")
