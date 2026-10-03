from copy import copy
from pathlib import Path

from docx import Document


SOURCE = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_pagina3_file.docx")
OUTPUT = Path(r"D:\TCCIaBigData\TCC_ResNet_FINAL_revisado_ABNT_pagina3_file_REVISADO.docx")


REPLACEMENTS = {
    21: "\t\t\t\t\t\tOrientador: Prof. Francisco Aparecido Rodrigues.",
    75: "5 CONCLUSÃO\t19",
    76: "REFERÊNCIAS\t20",
    77: "APÊNDICE A – DECLARAÇÃO DE USO DE INTELIGÊNCIA ARTIFICIAL\t22",
    108: (
        "O aumento da profundidade de uma rede neural não garante melhor desempenho e pode "
        "dificultar a otimização. A ResNet reformula blocos de camadas para aprender uma função "
        "residual F(x), cuja saída é somada à entrada x, resultando em H(x) = F(x) + x. As "
        "conexões de identidade oferecem um caminho direto para a propagação da informação e dos "
        "gradientes, facilitando o treinamento de redes profundas (HE et al., 2016)."
    ),
    109: (
        "As variantes ResNet18 e ResNet34 utilizam blocos residuais básicos, enquanto a ResNet50 "
        "emprega blocos bottleneck, o que altera o número de parâmetros e o custo computacional."
    ),
    112: (
        "\tO ImageNet foi proposto como uma base hierárquica de imagens em larga escala para apoiar "
        "pesquisas de reconhecimento visual (DENG et al., 2009). As representações aprendidas nessa "
        "base podem servir como inicialização para outra tarefa. Essa estratégia, denominada "
        "aprendizado por transferência, aproveita informações visuais gerais de um conjunto para "
        "reduzir o esforço necessário ao aprendizado em um conjunto menor."
    ),
    113: (
        "\tNeste estudo, o aprendizado por transferência foi avaliado por meio de três configurações: "
        "ajuste completo dos pesos pré-treinados, treinamento da ResNet18 a partir de uma "
        "inicialização aleatória e uso da ResNet18 pré-treinada com o backbone congelado. Isso "
        "permitiu comparar as configurações sob condições equivalentes."
    ),
    116: (
        "\tO EuroSAT reúne 27.000 imagens rotuladas em dez classes e foi construído a partir de dados "
        "do satélite Sentinel-2 (HELBER et al., 2019). Existem duas versões do EuroSAT: a "
        "multiespectral, que utiliza 13 bandas do Sentinel-2, e a RGB, que utiliza somente as bandas "
        "visíveis correspondentes ao vermelho, ao verde e ao azul, armazenadas como imagens JPEG. "
        "Escolhemos a versão RGB por sua compatibilidade com os modelos pré-treinados no ImageNet e "
        "para facilitar a reprodução do experimento, embora as bandas adicionais da versão "
        "multiespectral pudessem fornecer informações complementares."
    ),
    129: (
        "\tA acurácia corresponde à proporção de classificações corretas. A precisão, o recall e o "
        "F1 foram calculados por classe. O macro-F1 é a média não ponderada dos valores de F1 das dez "
        "classes, independentemente da quantidade de imagens em cada uma. Para avaliar a eficiência, "
        "consideramos a latência, o throughput, o pico de memória e o tamanho do checkpoint."
    ),
    138: (
        "\tA pesquisa é quantitativa e experimental e compara arquiteturas ResNet em diferentes "
        "condições de treinamento, incluindo pré-treinamento e congelamento do backbone. Os "
        "resultados são expressos por métricas preditivas e medidas de custo computacional."
    ),
    141: (
        "\tA preparação dos dados começou com a divisão das imagens por classe, gerada uma única vez "
        "com a semente 42. Os índices foram salvos em um arquivo JSON e reutilizados em todos os "
        "experimentos para evitar variações decorrentes de mudanças na composição das partições de "
        "treinamento e validação. As imagens da partição de teste não foram utilizadas até a escolha "
        "do modelo a ser avaliado."
    ),
    153: (
        "\tO checkpoint de cada execução foi determinado pela menor perda de validação. A semente de "
        "treinamento controlou a aleatoriedade dos dados e da inicialização, enquanto a semente da "
        "divisão permaneceu fixa. Para cada arquitetura, foram realizadas três execuções com sementes "
        "distintas, o que permitiu calcular a média e o desvio padrão dos resultados."
    ),
    156: (
        "\tO baseline ingênuo sempre prevê a classe mais frequente no conjunto de treinamento. Ele "
        "estabelece um limite inferior, mas não utiliza as características das imagens. A ResNet18 "
        "treinada do zero mede o desempenho sem aprendizado por transferência. As execuções com o "
        "backbone congelado avaliam uma alternativa de menor custo de ajuste, pois treinam apenas a "
        "camada final."
    ),
    165: (
        "Antes de iniciar o teste, definimos uma regra para selecionar a arquitetura. Apenas as "
        "arquiteturas situadas até 0,5 ponto percentual abaixo do melhor macro-F1 médio na validação "
        "seriam elegíveis; entre elas, seria selecionada aquela com a menor latência média para "
        "tamanho de lote 1. Para a avaliação final, seria utilizado o checkpoint da semente 42. Essas "
        "decisões foram registradas em um arquivo antes da avaliação final, e o teste foi executado "
        "uma única vez, sem alterações de arquitetura, semente ou hiperparâmetros."
    ),
    172: (
        "\tOs resultados mostram que todas as redes superaram amplamente o baseline majoritário, o "
        "que indica que os modelos aprenderam padrões discriminativos. A ResNet18 treinada do zero "
        "obteve macro-F1 de 96,13%, cerca de 2,30 pontos percentuais abaixo da versão pré-treinada. "
        "Esse resultado evidencia a contribuição do aprendizado por transferência no protocolo "
        "adotado."
    ),
    173: (
        "\tAo congelar os pesos da ResNet18 e aplicar diferentes taxas de aprendizado às camadas "
        "treináveis, observamos perda de desempenho com uma taxa de aprendizado (lr) de 10⁻⁴. Com o "
        "aumento para lr = 10⁻³, o macro-F1 chegou a 90,08%, mas permaneceu bem abaixo do resultado "
        "obtido com o ajuste completo. Portanto, a economia de treinamento proporcionada pelo "
        "congelamento reduziu de forma relevante o desempenho preditivo do modelo."
    ),
    182: (
        "A ResNet34 apresentou macro-F1 médio 0,0341 ponto percentual superior ao da ResNet18, mas "
        "sua latência foi aproximadamente 1,86 vez maior. Portanto, o ganho não compensou o custo "
        "computacional adicional. Os baixos desvios padrão observados sugerem pouca variação entre as "
        "três sementes, dentro da divisão de dados utilizada."
    ),
    193: (
        "Avaliamos uma única vez o checkpoint da ResNet18 pré-treinada com a semente 42 nas 4.050 "
        "imagens de teste. O modelo obteve acurácia de 98,6667% e macro-F1 de 98,6122%. Em comparação "
        "com a validação do mesmo checkpoint, o teste final apresentou acréscimos de cerca de 0,173 "
        "ponto percentual na acurácia e 0,155 ponto percentual no macro-F1. Os resultados foram "
        "semelhantes nas partições de validação e teste do EuroSAT."
    ),
    211: (
        "O cultivo permanente apresentou o menor F1 entre as classes, de 97,30%. O resultado decorreu "
        "da classificação incorreta de algumas imagens dessa classe: oito foram classificadas como "
        "cultivo anual e seis como vegetação herbácea. Essas confusões podem estar relacionadas às "
        "semelhanças de textura, cor e organização espacial entre áreas agrícolas."
    ),
    212: (
        "As classes com os maiores valores de F1 foram Mar/Lago e Floresta, com 99,78% e 99,67%, "
        "respectivamente. A matriz de confusão registrou poucos erros nessas duas classes, em "
        "consonância com os valores de F1 observados."
    ),
    215: (
        "\tO experimento apresenta limitações e ameaças à validade. Os resultados são específicos do "
        "EuroSAT RGB e, portanto, podem não se reproduzir em imagens de outros satélites, resoluções, "
        "regiões geográficas ou períodos do ano. Além disso, o uso de apenas três sementes não elimina "
        "a incerteza associada ao treinamento. O benchmark foi realizado em uma única configuração de "
        "hardware, e seus resultados podem variar conforme o dispositivo. A comparação também se "
        "restringiu à família ResNet; trabalhos futuros podem incluir outras famílias de arquiteturas. "
        "Por fim, como o EuroSAT é composto por recortes de imagens, os resultados não devem ser "
        "extrapolados diretamente para tarefas de mapeamento contínuo."
    ),
    219: (
        "Entre as três arquiteturas, a ResNet50 obteve o maior macro-F1 na validação, de 98,8644%, mas "
        "apresentou maior latência, uso de memória e demanda de armazenamento. A ResNet18 apresentou "
        "a melhor relação custo-benefício, com macro-F1 de 98,5113% e menor latência. No teste final, "
        "obteve acurácia de 98,6667% e macro-F1 de 98,6122%."
    ),
    220: (
        "Os experimentos permitiram alcançar o objetivo principal. Nas condições avaliadas, a "
        "ResNet18 pré-treinada e integralmente ajustada apresentou a melhor relação entre desempenho e "
        "custo computacional. A ResNet50 obteve uma pequena vantagem preditiva, acompanhada por "
        "aumentos de latência, memória e armazenamento. A escolha, contudo, depende da aplicação: se "
        "a prioridade for maximizar o macro-F1, mesmo com maior custo computacional, a ResNet50 "
        "permanece viável; quando a eficiência for prioritária, recomenda-se a ResNet18."
    ),
    221: (
        "Como trabalhos futuros, podem-se testar imagens de outras regiões, bandas multiespectrais e "
        "arquiteturas distintas. Também é relevante investigar a calibração e a incerteza das "
        "previsões, principalmente nas classes agrícolas que apresentaram as maiores taxas de erro."
    ),
    283: (
        "Declaro, ainda, que compreendo plenamente, sou capaz de explicar e defender todo o conteúdo "
        "apresentado neste TCC, bem como observei os princípios éticos e os requisitos de proteção de "
        "dados aplicáveis."
    ),
}


def replace_paragraph_text(paragraph, new_text):
    """Replace text while preserving paragraph properties and the first run's formatting."""
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return

    first = paragraph.runs[0]
    first.text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""


def main():
    doc = Document(SOURCE)

    for index, new_text in REPLACEMENTS.items():
        replace_paragraph_text(doc.paragraphs[index], new_text)

    # Padronização pontual da terminologia na tabela de resultados, sem alterar
    # estrutura, dimensões ou formatação da tabela.
    results = doc.tables[2]
    results.cell(1, 0).paragraphs[0].runs[0].text = "Baseline majoritário"
    results.cell(2, 0).paragraphs[0].runs[0].text = "ResNet18 congelada, lr = 10⁻⁴"
    results.cell(3, 0).paragraphs[0].runs[0].text = "ResNet18 congelada, lr = 10⁻³"
    results.cell(5, 0).paragraphs[0].runs[0].text = "ResNet18 pré-treinada"

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
