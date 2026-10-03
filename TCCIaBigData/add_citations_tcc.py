from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement


SOURCE = Path(
    r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_pagina3_file_REVISADO.docx"
)
OUTPUT = Path(
    r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_pagina3_file_REVISADO_CITACOES.docx"
)


def paragraph_starting_with(document, prefix):
    for paragraph in document.paragraphs:
        if paragraph.text.startswith(prefix):
            return paragraph
    raise ValueError(f"Parágrafo não encontrado: {prefix}")


def replace_once(paragraph, old, new):
    if old not in paragraph.text:
        raise ValueError(f"Trecho não encontrado em: {paragraph.text}")

    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new, 1)
            return

    # Os parágrafos-alvo possuem um único run no arquivo atual. Esta alternativa
    # mantém as propriedades do primeiro run caso o Word os fragmente no futuro.
    full_text = paragraph.text.replace(old, new, 1)
    paragraph.runs[0].text = full_text
    for run in paragraph.runs[1:]:
        run.text = ""


def insert_reference_after(paragraph, text):
    new_p = OxmlElement("w:p")
    if paragraph._p.pPr is not None:
        new_p.append(deepcopy(paragraph._p.pPr))

    new_r = OxmlElement("w:r")
    if paragraph.runs and paragraph.runs[0]._r.rPr is not None:
        new_r.append(deepcopy(paragraph.runs[0]._r.rPr))
    new_t = OxmlElement("w:t")
    new_t.text = text
    new_r.append(new_t)
    new_p.append(new_r)

    paragraph._p.addnext(new_p)

    # Retorna um objeto Paragraph válido para permitir inserções encadeadas.
    from docx.text.paragraph import Paragraph

    return Paragraph(new_p, paragraph._parent)


def main():
    doc = Document(SOURCE)

    intro = paragraph_starting_with(doc, "Nos últimos anos")
    replace_once(
        intro,
        "auxiliando a triagem, o mapeamento e o monitoramento.",
        "auxiliando a triagem, o mapeamento e o monitoramento (ZHU et al., 2017).",
    )

    cnn = paragraph_starting_with(doc, "\tA classificação de imagens consiste")
    replace_once(
        cnn,
        "Ao final, as camadas de classificação produzem escores para cada classe.",
        "Ao final, as camadas de classificação produzem escores para cada classe "
        "(GOODFELLOW; BENGIO; COURVILLE, 2016).",
    )

    transfer = paragraph_starting_with(doc, "\tO ImageNet foi proposto")
    replace_once(
        transfer,
        "reduzir o esforço necessário ao aprendizado em um conjunto menor.",
        "reduzir o esforço necessário ao aprendizado em um conjunto menor "
        "(PAN; YANG, 2010).",
    )

    metrics = paragraph_starting_with(doc, "\tA acurácia corresponde")
    replace_once(
        metrics,
        "independentemente da quantidade de imagens em cada uma.",
        "independentemente da quantidade de imagens em cada uma "
        "(SOKOLOVA; LAPALME, 2009).",
    )

    deng = paragraph_starting_with(doc, "DENG, J.")
    insert_reference_after(
        deng,
        "GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. Deep learning. Cambridge, MA: "
        "MIT Press, 2016.",
    )

    helber = paragraph_starting_with(doc, "HELBER, P.")
    insert_reference_after(
        helber,
        "PAN, S. J.; YANG, Q. A survey on transfer learning. IEEE Transactions on "
        "Knowledge and Data Engineering, v. 22, n. 10, p. 1345-1359, 2010. DOI: "
        "10.1109/TKDE.2009.191.",
    )

    paszke = paragraph_starting_with(doc, "PASZKE, A.")
    sokolova = insert_reference_after(
        paszke,
        "SOKOLOVA, M.; LAPALME, G. A systematic analysis of performance measures for "
        "classification tasks. Information Processing & Management, v. 45, n. 4, "
        "p. 427-437, 2009. DOI: 10.1016/j.ipm.2009.03.002.",
    )
    insert_reference_after(
        sokolova,
        "ZHU, X. X. et al. Deep learning in remote sensing: a comprehensive review and "
        "list of resources. IEEE Geoscience and Remote Sensing Magazine, v. 5, n. 4, "
        "p. 8-36, 2017. DOI: 10.1109/MGRS.2017.2762307.",
    )

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
