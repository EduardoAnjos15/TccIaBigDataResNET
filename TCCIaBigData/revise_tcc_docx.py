from copy import deepcopy
from pathlib import Path
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


SOURCE = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])


REPLACEMENTS = {
    "Este Trabalho tem por objetivo investigar a relação entre desempenho preditivo e custo computacional de diferentes arquiteturas de ResNet aplicadas à classificação de uso e cobertura do solo em imagens de satélite. Foram comparadas arquiteturas ResNet18, ResNet34 e ResNet50 no conjunto EuroSAT RGB, que é composto por 27.000 imagens distribuídas em 10 classes. Os experimentos utilizaram uma divisão fixa de 70% para treinamento, 15% para validação e 15% para teste, aprendizado por transferência a partir do ImageNet e 3 seeds de treinamento. As principais medições foram de acurácia, macro-F1, latência, throughput, memória de inferência e tamanho do checkpoint. Sobre os Resultados, a arquitetura que obteve o melhor macro-F1 médio Foi a ResNet50 que obteve o resultado de 98,8644%, a ResNet18 obteve o resultado de 98,5113%, com uma latência aproximadamente 2,56 vezes menos em interferência unitária. Implementando a regra de seleção registrada anteriormente no teste. A ResNet18 foi escolhida como o melhor custo-benefício entre as arquiteturas. O último teste foi realizado apenas uma vez em uma amostra de 4.050 imagens, o modelo teve uma acurácia de 98,6667% e um macro-F1 de 98,6122%. Os resultados demonstram que a arquitetura ResNet18 oferece um alto desempenho com um custo computacional menor no cenário avaliado.":
    "Este trabalho tem como objetivo investigar a relação entre o desempenho preditivo e o custo computacional de diferentes arquiteturas ResNet aplicadas à classificação do uso e da cobertura do solo em imagens de satélite. Foram comparadas as arquiteturas ResNet18, ResNet34 e ResNet50 no conjunto EuroSAT RGB, composto por 27.000 imagens distribuídas em 10 classes. Os experimentos utilizaram uma divisão fixa de 70% para treinamento, 15% para validação e 15% para teste, aprendizado por transferência a partir do ImageNet e três sementes de treinamento. As principais métricas foram acurácia, macro-F1, latência, throughput, memória de inferência e tamanho do checkpoint. A ResNet50 obteve o maior macro-F1 médio, de 98,8644%, enquanto a ResNet18 alcançou 98,5113%; a latência da ResNet50 na inferência unitária foi aproximadamente 2,56 vezes a da ResNet18. De acordo com a regra de seleção definida antes do teste, a ResNet18 foi escolhida por apresentar a melhor relação custo-benefício. O teste final foi realizado uma única vez em uma amostra de 4.050 imagens, na qual o modelo obteve acurácia de 98,6667% e macro-F1 de 98,6122%. Os resultados indicam que, no cenário avaliado, a ResNet18 oferece alto desempenho com menor custo computacional.",
    "Palavras-chave: classificação de imagens. ResNet. EuroSAT. sensores remotos. transferência de aprendizado. custo-benefício.":
    "Palavras-chave: classificação de imagens; ResNet; EuroSAT; sensoriamento remoto; aprendizado por transferência; custo-benefício.",
    "This study aims to investigate the relationship between predictive performance and computational cost among different ResNet architectures applied to land-use and land-cover classification in satellite imagery. The ResNet18, ResNet34, and ResNet50 architectures were compared using the EuroSAT RGB dataset, which comprises 27,000 images distributed across 10 classes. The experiments employed a fixed split of 70% for training, 15% for validation, and 15% for testing, transfer learning from ImageNet, and three training seeds. The main metrics evaluated were accuracy, macro-F1, latency, throughput, inference memory usage, and checkpoint size. ResNet50 achieved the highest mean macro-F1 score, at 98.8644%, whereas ResNet18 achieved 98.5113%, with approximately 2.56 times lower latency for single-image inference. Based on the selection rule established before testing, ResNet18 was chosen as the architecture offering the best cost–benefit trade-off. The final test was conducted only once on a sample of 4,050 images, in which the model achieved 98.6667% accuracy and a macro-F1 score of 98.6122%. The results demonstrate that the ResNet18 architecture provides high performance at a lower computational cost in the evaluated scenario.":
    "This study investigates the relationship between predictive performance and computational cost among ResNet architectures applied to land-use and land-cover classification in satellite imagery. ResNet18, ResNet34, and ResNet50 were compared using the EuroSAT RGB dataset, which comprises 27,000 images distributed across 10 classes. The experiments used a fixed split of 70% for training, 15% for validation, and 15% for testing, transfer learning from ImageNet, and three training seeds. The evaluated metrics were accuracy, macro-F1, latency, throughput, inference memory usage, and checkpoint size. ResNet50 achieved the highest mean macro-F1 score, at 98.8644%, whereas ResNet18 achieved 98.5113%; ResNet50's single-image inference latency was approximately 2.56 times that of ResNet18. Based on the selection rule established before testing, ResNet18 was chosen as the architecture offering the best cost-benefit trade-off. The final test was conducted once on 4,050 images, and the model achieved 98.6667% accuracy and a macro-F1 score of 98.6122%. The results indicate that ResNet18 provides high performance at a lower computational cost in the evaluated scenario.",
    "Keywords: image classification. ResNet. EuroSAT. remote sensing. transfer learning. cost–benefit trade-off.":
    "Keywords: image classification; ResNet; EuroSAT; remote sensing; transfer learning; cost-benefit trade-off.",
    "Nos últimos anos, o uso de imagens via satélite aumentou significativamente a capacidade de observar o nosso planeta, acompanhar grandes áreas e analisar as transformações que acontecem no território. Entretanto, por conta da grande extensão territorial e por conta do massivo volume de dados produzidos por programas de observação da terra torna inviável a inspeção exclusivamente manual. Tecnologias de reconhecimento e classificação de imagens podem apoiar esse processo atribuindo elas a uma categoria de uso ou cobertura de solo, fornecendo um apoio para triagem, mapeamento e monitoramento.":
    "Nos últimos anos, o uso de imagens de satélite ampliou significativamente a capacidade de observar o planeta, acompanhar grandes áreas e analisar as transformações do território. Entretanto, a extensão territorial e o grande volume de dados produzidos por programas de observação da Terra tornam inviável a inspeção exclusivamente manual. Tecnologias de reconhecimento e classificação de imagens podem apoiar esse processo ao atribuí-las a categorias de uso ou cobertura do solo, auxiliando a triagem, o mapeamento e o monitoramento.",
    "As redes neurais convolucionais são altamente citadas quando se trata de classificação de imagens, o principal motivo é por aprenderem diretamente dos pixels. Entre essas redes, a Residual Network (ResNet) introduziu conexões de atalho que favorecem a otimização de modelos profundos (HE et al., 2016). Diferentes profundidades, porém, há combinações diferentes relacionadas a desempenho, latência, tamanho de armazenamento e memória. No mundo real, onde o custo computacional cresce de forma relevante ao, utilizar o modelo com a maior acurácia nem sempre é a melhor opção.":
    "As redes neurais convolucionais são amplamente utilizadas na classificação de imagens porque aprendem diretamente a partir dos pixels. Entre essas redes, a Residual Network (ResNet) introduziu conexões de atalho que favorecem a otimização de modelos profundos (HE et al., 2016). Diferentes profundidades, porém, produzem combinações distintas de desempenho, latência, tamanho de armazenamento e uso de memória. Em aplicações reais, nas quais o custo computacional é relevante, utilizar o modelo de maior acurácia nem sempre é a melhor opção.",
    "Esse trabalho delimita o problema à classificação de imagens RGB do EuroSAT em dez classes: floresta, cultivo anual, vegetação herbácea, rodovia, área industrial, pastagem, cultivo permanente, área residencial, rio e mar/lago. O conjunto foi derivado de imagens Sentinel-2 e criado para pesquisas de uso e cobertura do solo (HELBER et al., 2019).":
    "Este trabalho delimita o problema à classificação de imagens RGB do EuroSAT em dez classes: floresta, cultivo anual, vegetação herbácea, rodovia, área industrial, pastagem, cultivo permanente, área residencial, rio e mar/lago. O conjunto foi derivado de imagens do Sentinel-2 e criado para pesquisas sobre uso e cobertura do solo (HELBER et al., 2019).",
    "O problema a ser resolvido consiste em selecionar uma arquitetura ResNet capaz de manter um desempenho de classificação alto com um custo computacional compatível com uma futura aplicação de análise de imagens de satélite. Basicamente trata-se de analisar as arquiteturas ResNet18, ResNet34 e ResNet50, e verificar qual apresenta a melhor relação entre o macro-F1 e o custo computacional na classificação das imagens do EuroSAT RGB sob um protocolo experimental controlado.":
    "O problema consiste em selecionar uma arquitetura ResNet capaz de manter alto desempenho de classificação com custo computacional compatível com uma futura aplicação de análise de imagens de satélite. Para isso, analisam-se as arquiteturas ResNet18, ResNet34 e ResNet50 a fim de verificar qual apresenta a melhor relação entre macro-F1 e custo computacional na classificação das imagens do EuroSAT RGB, sob um protocolo experimental controlado.",
    "O objetivo geral é comparar as três arquiteturas ResNet na classificação das imagens  de uso e cobertura do solo em imagens do EuroSAT. Para alcançar o objetivo geral, foram definidos os seguintes objetivos específicos:":
    "O objetivo geral é comparar três arquiteturas ResNet na classificação de imagens de uso e cobertura do solo do EuroSAT. Para alcançar esse objetivo, foram definidos os seguintes objetivos específicos:",
    "a) Implementar um pipeline de treinamento teste e validação utilizando o PyTorch;": "a) Implementar um pipeline de treinamento, validação e teste utilizando o PyTorch;",
    "c) Avaliar o efeito da transferência de aprendizado e do congelamento do extrator de caracteristicas;": "c) Avaliar o efeito do aprendizado por transferência e do congelamento do extrator de características;",
    "f) Analisar o desempenho final do modelo selecionados;": "f) Analisar o desempenho final do modelo selecionado.",
    "Além desta introdução abordaremos os conceitos de classificação de imagens redes convolucionais, aprendizado residual, ImageNet, transferência de aprendizado e EuroSAT no Capítulo 2. O protocolo realizado pelo experimento é descrito no Capítulo 3. A discussão dos resultados no Capítulo 4 e as conclusões, limitações e possibilidades de evolução são discutidas no Capítulo 5.":
    "Além desta introdução, o Capítulo 2 aborda os conceitos de classificação de imagens, redes convolucionais, aprendizado residual, ImageNet, aprendizado por transferência e EuroSAT. O Capítulo 3 descreve o protocolo experimental; o Capítulo 4 apresenta e discute os resultados; e o Capítulo 5 reúne as conclusões, as limitações e as possibilidades de trabalhos futuros.",
    "2.3 ImageNet e transferência de aprendizado": "2.3 ImageNet e aprendizado por transferência",
    "Classificação de imagens consiste em associar determinada imagem a uma categoria ou grupo previamente definido. Quando falamos de redes neurais convolucionais, existem pequenos filtros que identificam determinados padrões como bordas, formas e texturas. Quanto mais a informação atravessa esses filtros, esses padrões são combinados. Ao final, essas camadas de classificação produz escores para cada classe.":
    "A classificação de imagens consiste em associar determinada imagem a uma categoria ou grupo previamente definido. Nas redes neurais convolucionais, pequenos filtros identificam padrões como bordas, formas e texturas. À medida que a informação atravessa esses filtros, os padrões são combinados. Ao final, as camadas de classificação produzem escores para cada classe.",
    "Quando trazemos o senário para o mundo real sabemos que a aparência de imagens de uma determinada classe pode ser influenciada por condições atmosféricas, sazonalidade, resolução espacial e características regionais. Imagens relacionadas a áreas agrícolas porém de classes diferentes podem compartilhar certas cores e texturas, enquanto imagens de florestas e mar apresentam padrões mais distintos. Por esse motivo, executar a análise por classe torna a acurácia global mais assertiva.":
    "Em cenários reais, a aparência das imagens de determinada classe pode ser influenciada por condições atmosféricas, sazonalidade, resolução espacial e características regionais. Imagens de áreas agrícolas pertencentes a classes diferentes podem compartilhar cores e texturas, enquanto imagens de florestas e de mar apresentam padrões mais distintos. Por esse motivo, a análise por classe complementa a acurácia global e torna a avaliação mais informativa.",
    "Utilizando o mesmo conjunto de dados analisaremos particularidades das arquiteturas ResNet, as variantes da arquitetura ResNet18 e ResNet34 utilizam blocos residuais mais basicos enquanto a ResNet50 usa blocos bottleneck alterando o número de parâmetros e custo.":
    "Com o mesmo conjunto de dados, analisamos as particularidades das arquiteturas ResNet. As variantes ResNet18 e ResNet34 utilizam blocos residuais mais básicos, enquanto a ResNet50 emprega blocos bottleneck, o que altera o número de parâmetros e o custo computacional.",
    "O ImageNet foi proposto como uma base hierárquica de imagens em larga escala para apoia pesquisas de reconhecimento visual (DENG et al., 2009). As informações aprendidas nessa base pode servir como um inicializador para uma outra tarefa. Essa estratégia é denominada transferência de aprendizado e consiste em aproveitar informações visuais gerais de um conjunto para reduzir o esforço necessário para o aprendizado a partir de um conjunto menor.":
    "O ImageNet foi proposto como uma base hierárquica de imagens em larga escala para apoiar pesquisas de reconhecimento visual (DENG et al., 2009). As informações aprendidas nessa base podem servir como inicialização para outra tarefa. Essa estratégia, denominada aprendizado por transferência, aproveita informações visuais gerais de um conjunto para reduzir o esforço necessário ao aprendizado em um conjunto menor.",
    "O objetivo de também utilizar a transferência de aprendizado nesse estudo é medir seus efeitos e também o efeito da arquitetura isoladamente. Para isso, utilizamos três configurações: ajuste total dos valores pré treinados, treinamento a partir de uma inicialização aleatória da ResNet18 e uso da ResNet18 pré treinada com freeze backbone.":
    "O uso do aprendizado por transferência neste estudo permite medir seus efeitos e avaliar isoladamente o efeito da arquitetura. Para isso, utilizamos três configurações: ajuste completo dos pesos pré-treinados, treinamento da ResNet18 a partir de uma inicialização aleatória e uso da ResNet18 pré-treinada com o backbone congelado.",
    "O EuroSAT reúne 27000 imagens rotuladas em dez classes e foi construído a partir de dados do satélite Sentinel-2 (HELBER et al., 2019). Existem duas versões do EuroSAT, a multiespectral, que utiliza cerca de 13 bandas do Sentinel-2, e o RGB que utiliza somente as bandas visíveis vermelho, verde e azul, todas armazenadas como imagens JPEG. A decisão de escolhermos o RGB para a nossa pesquisa foi por conta da sua compatibilidade com os modelos pré treinados do ImageNet e também para deixar o nosso experimento um pouco mais simples de ser reproduzido, embora as bandas adicionais do multiespectral poderia ter nos fornecer informações adicionais.":
    "O EuroSAT reúne 27.000 imagens rotuladas em dez classes e foi construído a partir de dados do satélite Sentinel-2 (HELBER et al., 2019). Existem duas versões do EuroSAT: a multiespectral, que utiliza 13 bandas do Sentinel-2, e a RGB, que utiliza somente as bandas visíveis vermelha, verde e azul, armazenadas como imagens JPEG. Escolhemos a versão RGB por sua compatibilidade com os modelos pré-treinados no ImageNet e para facilitar a reprodução do experimento, embora as bandas adicionais da versão multiespectral pudessem fornecer informações complementares.",
    "Figura 1 – Três exemplos de cada uma das 10 classes do EuroSAT utilizadas no estudo": "Figura 1 – Três exemplos de cada uma das dez classes do EuroSAT utilizadas no estudo",
    "Fonte: elaborado pelo autor através das imagens do EuroSAT.": "Fonte: elaborado pelo autor com imagens do EuroSAT.",
    "A acurácia se resume a porcentagem de classificações corretas no teste. A precisão, recall e F1 foram calculados por classe. O macro-F1 é a média não ponderada calculada nas dez classes, independente da quantidade de imagens de cada classe. Para testarmos a eficiência consideramos latência, throughput, pico de memória e tamanho do checkpoint.":
    "A acurácia corresponde à porcentagem de classificações corretas no teste. A precisão, o recall e o F1 foram calculados por classe. O macro-F1 é a média não ponderada das dez classes, independentemente da quantidade de imagens em cada uma. Para avaliar a eficiência, consideramos a latência, o throughput, o pico de memória e o tamanho do checkpoint.",
    "A pesquisa tem uma natureza quantitativa e experimental, com o foco principal nas arquiteturas ResNet, Também realizamos experimentos com pré treinamento e freeze do backbone. Os dados obtidos são medições de custo computacional e métricas preditivas.":
    "A pesquisa é de natureza quantitativa e experimental, com foco nas arquiteturas ResNet. Também realizamos experimentos com pré-treinamento e congelamento do backbone. Os dados obtidos consistem em métricas preditivas e medições de custo computacional.",
    "A preparação dos dados iniciou com uma divisão das imagens por classe e gerada uma única vez com seed 42. Os índices foram salvos em um arquivo JSON e foram reutilizados em todos os testes da pesquisa para não ocorrer variações de resultados por mudanças de imagens, tanto no treinamento quanto na validação. As imagens da divisão de teste não foram usadas até escolhermos o modelo a ser testado.":
    "A preparação dos dados começou com a divisão das imagens por classe, gerada uma única vez com a semente 42. Os índices foram salvos em um arquivo JSON e reutilizados em todos os experimentos para evitar variações decorrentes de mudanças nas imagens de treinamento e validação. As imagens da partição de teste não foram utilizadas até a escolha do modelo a ser avaliado.",
    "No treino aplicamo transformações como recortes aleatórios, redimensionamento para 224 x 224 pixels, inversões horizontais, verticais, rotações e normalização com média e desvio padrão do ImageNet. Nas partes de validação e teste, usamos imagens redimensionadas para 256 pixels e recortadas para 224 x 224 e normalizadas da mesma forma.":
    "No treinamento, aplicamos transformações como recortes aleatórios, redimensionamento para 224 × 224 pixels, inversões horizontais e verticais, rotações e normalização com a média e o desvio padrão do ImageNet. Na validação e no teste, redimensionamos as imagens para 256 pixels, recortamo-las para 224 × 224 pixels e aplicamos a mesma normalização.",
    "A implementação utilizou a linguagem de programação Python, com as bibliotecas PyTorch e torchvision. O PyTorch oferece execução eficiente em aceleradores e uma interface imperativa adequada à experimentação (PASZKE et al., 2019) De hardware utilizamos uma GPU NVIDIA GeForce RTX 3060 com 12GB.":
    "A implementação utilizou a linguagem de programação Python e as bibliotecas PyTorch e torchvision. O PyTorch oferece execução eficiente em aceleradores e uma interface imperativa adequada à experimentação (PASZKE et al., 2019). Como hardware, utilizamos uma GPU NVIDIA GeForce RTX 3060 com 12 GB de memória.",
    "O checkpoint de cada execução foi determinado pela menor perda da validação. A seed de treino controlou a aleatoriedade das informações e  a inicialização enquanto a seed de divisão permaneceu fixa. Para cada arquitetura foram executados 3 seeds, nos permitindo obter a média e o desvio padrão das amostras.":
    "O checkpoint de cada execução foi determinado pela menor perda de validação. A semente de treinamento controlou a aleatoriedade dos dados e da inicialização, enquanto a semente da divisão permaneceu fixa. Para cada arquitetura, foram realizadas três execuções com sementes distintas, o que permitiu calcular a média e o desvio padrão das amostras.",
    "O comportamento da base ingênua é sempre prever a classe mais frequente no treino, ela estabelece um limite inferior porém não utiliza as caracteristicas das imagens. A ResNet18 treinada do zero mede o desempenho sem a transferência de aprendizado. Quando fazemos as execuções com o freeze backbone podemos avaliar uma alternatica com um custo menor de ajuste e elas treinam apenas a camada final.":
    "A baseline ingênua sempre prevê a classe mais frequente no conjunto de treinamento. Ela estabelece um limite inferior, mas não utiliza as características das imagens. A ResNet18 treinada do zero mede o desempenho sem aprendizado por transferência. As execuções com o backbone congelado avaliam uma alternativa de menor custo de ajuste, pois treinam apenas a camada final.",
    "No passo da validação utilizamos os checkpoints da seed 42 no benchmark. E foram medidos os batch sizes de 1 e 64, com 20 lotes de aquecimento, cinco repetições e 100 forwards sincronizados por repetição por modelo. Utilizamos o tempo obtido pelos eventos CUDA. O fluxo completo incluiu a leitura, a transferência para a GPU a execução do modelo e as transformações. Para reduzir as possíveis interferências do enfileiramento assincrono utilizamos a memória principal para corresponder ao pico do forward sincronizado.":
    "Na validação, utilizamos no benchmark os checkpoints da semente 42. Foram avaliados tamanhos de lote de 1 e 64, com 20 lotes de aquecimento, cinco repetições e 100 passagens sincronizadas por repetição e por modelo. O tempo foi medido por meio de eventos CUDA. O fluxo completo incluiu a leitura, as transformações, a transferência para a GPU e a execução do modelo. Para reduzir interferências do enfileiramento assíncrono, sincronizamos a GPU; o uso de memória correspondeu ao pico observado durante a passagem sincronizada.",
    "Antes da inicialização do teste, estipulamos uma regra para melhor organizar as execuções, definimos que apenas as arquiteturas de até 0,5 pontos percentuais abaixo do melhor macro-F1 médio na validação seriam elegíveis, e entre estas, apenas a com a menor latência média do modelo com o batch size 1 seria selecionada. Para o checkpoint representativo utilizaríamos apenas o da seed 42. Essas decisões foram registradas em um arquivo antes da avaliação final e o teste foi executado apenas uma única vez e não alteramos a arquitetura, seed ou hiperparâmetros nele.":
    "Antes de iniciar o teste, definimos uma regra para organizar as execuções. Apenas as arquiteturas situadas até 0,5 ponto percentual abaixo do melhor macro-F1 médio na validação seriam elegíveis; entre elas, seria selecionada aquela com a menor latência média em lote unitário (batch size 1). O checkpoint representativo seria o da semente 42. Essas decisões foram registradas em um arquivo antes da avaliação final, e o teste foi executado uma única vez, sem alterações de arquitetura, semente ou hiperparâmetros.",
    "Analisando os resultados, podemos observar que todas as redes superaram em muito a baseline majoritária, isso significa que realmente os modelos aprenderam os padrões discriminativos. A ResNet18 treinada do zero obteve o macro-F1 de 96,13% mas ainda assim ficou a aproximadamente 2,30 pontos percentuais da ResNet18 da versão pré treinada. Podemos ver uma contribuição real da transferência de aprendizado no protocolo adotado.":
    "Os resultados mostram que todas as redes superaram amplamente a baseline majoritária, o que indica que os modelos aprenderam padrões discriminativos. A ResNet18 treinada do zero obteve macro-F1 de 96,13%, cerca de 2,30 pontos percentuais abaixo da versão pré-treinada. Esse resultado evidencia a contribuição do aprendizado por transferência no protocolo adotado.",
    "4.1 Baselines e efeito da transferência de aprendizado": "4.1 Baselines e efeito do aprendizado por transferência",
    "Quando congelamos os pesos do ResNet18 com learning rates diferentes nas camadas treináveis, observamos que a camada com lr= 10⁻⁴ obteve uma perda no desempenho, quando aumentamos para o  lr=10⁻³, o macro-F1 aumentou para 90,08% mas ainda assim ficou bem distante do ajuste completo. Portanto a economia com o treinamento obtida pelo congelamento acabou reduzindo muito o desempenho preditivo do modelo.":
    "Ao congelar os pesos da ResNet18 e aplicar diferentes taxas de aprendizado às camadas treináveis, observamos perda de desempenho com lr = 10⁻⁴. Com o aumento para lr = 10⁻³, o macro-F1 chegou a 90,08%, mas permaneceu bem abaixo do resultado obtido com o ajuste completo. Portanto, a economia de treinamento proporcionada pelo congelamento reduziu de forma relevante o desempenho preditivo do modelo.",
    "Figura 2 – Curvas de aprendizado da ResNet18 pré treinada, seed 42": "Figura 2 – Curvas de aprendizado da ResNet18 pré-treinada, semente 42",
    "A arquitetura ResNet50 obteve o melhor desempenho entre os modelos, tendo uma vantagem de aproximadamente 0,353% sobre a ResNet18, porém, a latência em batch 1 foi cerca de 2,56 vezes maior, o checkpoint foi aproximadamente 2,11 vezes maior e o throughput que seria em batch 64 correspondeu a aproximadamente 32,2% do throughput da ResNet18, ou seja, foi cerca de 67,8% menor.":
    "A ResNet50 obteve o melhor desempenho entre os modelos, com vantagem de aproximadamente 0,3531 ponto percentual sobre a ResNet18. Entretanto, sua latência em lote unitário foi cerca de 2,56 vezes maior, seu checkpoint foi aproximadamente 2,11 vezes maior e seu throughput em lote de 64 imagens correspondeu a 32,2% do throughput da ResNet18, ou seja, foi cerca de 67,8% menor.",
    "A arquitetura ResNet34 apresentou macro-F1 médio de 0,034% superior, mas com latência de aproximadamente 1,86 vezes maior, ou seja, o ganho não foi suficiente para compensar o custo computacional adicional. O desvio padrão mostrou pouca variação entre os seeds nas três arquiteturas, ou seja, obtivemos bastante estabilidade da comparação com o split utilizado.":
    "A ResNet34 apresentou macro-F1 médio 0,0341 ponto percentual superior ao da ResNet18, mas sua latência foi aproximadamente 1,86 vezes maior. Portanto, o ganho não compensou o custo computacional adicional. O desvio padrão indicou pouca variação entre as sementes nas três arquiteturas, o que demonstra estabilidade na comparação com a divisão utilizada.",
    "Figura 3 - Relações de Pareto entre macro-F1 e métricas de custo": "Figura 3 – Relações de Pareto entre macro-F1 e métricas de custo",
    "Todas as três arquiteturas ficaram dentro da margem de 0,5% definida anteriormente. A segunda etapa da regra nos fez selecionar a arquitetura ResNet18 por apresentar uma latência menor. A decisão de selecionar esta arquitetura específica não significa que ela é melhor, significa apenas que ela é um ótimo custo-benefício, temos uma arquitetura com uma eficiência maior e com o desempenho próximo do melhor valor observado.":
    "As três arquiteturas ficaram dentro da margem de 0,5 ponto percentual definida anteriormente. Na segunda etapa da regra, selecionamos a ResNet18 por apresentar a menor latência. Essa escolha não significa que a arquitetura seja a melhor em todos os critérios, mas que apresentou a melhor relação custo-benefício, combinando maior eficiência com desempenho próximo ao melhor valor observado.",
    "Avaliamos apenas uma vez o checkpoint da ResNet18 pré-treinada com o seed 42 nas 4.050 imagens de teste, e ela teve uma acurácia de 98,6667% com um macro-F1 de 98,6122%. Quando comparamos com a etapa de Validação do mesmo checkpoint, houve uma pequena diferença de cerca de 0,173% maior em acurácia e 0,155% maior em macro-F1 na etapa de teste final. Esta pequena diferença indica que houve uma boa generalização dentro da distribuição do EuroSAT.":
    "Avaliamos uma única vez o checkpoint da ResNet18 pré-treinada com a semente 42 nas 4.050 imagens de teste. O modelo obteve acurácia de 98,6667% e macro-F1 de 98,6122%. Em comparação com a validação do mesmo checkpoint, o teste final apresentou acréscimos de cerca de 0,173 ponto percentual na acurácia e 0,155 ponto percentual no macro-F1. Essa pequena diferença indica boa generalização dentro da distribuição do EuroSAT.",
    "Figura 4 - Comparação global entre validação e teste": "Figura 4 – Comparação global entre validação e teste",
    "Figura 5 - Matriz de confusão normalizada no teste final": "Figura 5 – Matriz de confusão normalizada no teste final",
    "Figura 6 - F1-score por classe no teste final": "Figura 6 – F1-score por classe no teste final",
    "Tabela 5 - Métricas por classe no teste final": "Tabela 5 – Métricas por classe no teste final",
    "O cultivo permanente apresentou o menor F1 entre todas as classes, de 97,30%. Esta porcentagem menor foi por conta de uma confusão de classificação de algumas imagens desta classe, foram oito classificadas como cultivo anual e seis classificadas como vegetação herbácea. Como existe uma semelhança entre texturas, cores e organização espacial entre essas áreas agrícolas, esses erros de classificação são coerentes.":
    "O cultivo permanente apresentou o menor F1 entre as classes, de 97,30%. O resultado decorreu da classificação incorreta de algumas imagens dessa classe: oito foram classificadas como cultivo anual e seis como vegetação herbácea. Como essas áreas agrícolas apresentam semelhanças de textura, cor e organização espacial, tais erros de classificação são coerentes.",
    "As classes que obtiveram os maiores valores de F1 foram Mar e Lago, com respectivamente 99,78% e 99,67%. Estas classes possuem características mais estáveis, o que explica esse F1 ligeiramente maior.":
    "As classes com os maiores valores de F1 foram Mar/Lago e Floresta, com 99,78% e 99,67%, respectivamente. Essas classes apresentam características mais estáveis, o que ajuda a explicar os valores ligeiramente superiores.",
    "Existem algumas limitações e ameaças a validade que podemos pontuar que estão relacionadas a este experimento, entre eles temos o fato de que os resultados são inerentes ao EuroSAT RGB, portanto não temos a garantia de que terá a mesma autonomia para outros satélites, resoluções, regiões geográficas ou períodos do ano. Podemos pontuar também baixo uso de apenas 3 seeds não elimina a incerteza associada ao treinamento. Quando falamos de benchmark, também temos o fato de que ele foi realizado em apenas uma configuração de hardware, ele pode mudar dependendo do dispositivo. Sobre a comparação, ela foi limitada à família ResNet por conta da sua simplicidade, um ponto de evolução seria comparar arquiteturas mais recentes ou outros modelos, e em relação ao EuroSAT, as suas imagens são recortes, o estudo não avalia mapas contínuos, as diferentes aplicações podem gerar diferentes resultados.":
    "O experimento apresenta limitações e ameaças à validade. Os resultados são específicos do EuroSAT RGB e, portanto, podem não se reproduzir em imagens de outros satélites, resoluções, regiões geográficas ou períodos do ano. Além disso, o uso de apenas três sementes não elimina a incerteza associada ao treinamento. O benchmark foi realizado em uma única configuração de hardware, e seus resultados podem variar conforme o dispositivo. A comparação também se restringiu à família ResNet; trabalhos futuros podem incluir arquiteturas mais recentes e outros modelos. Por fim, como o EuroSAT é composto por recortes de imagens, este estudo não avalia mapas contínuos, e outras aplicações podem produzir resultados diferentes.",
    "Este estudo comparou três arquiteturas da família ResNet, a ResNet18, a Resnet34 e a Resnet50 para a classificação de imagens via satélite para classificação de uso e cobertura do solo, retirados do conjunto EuroSat RGB. Utilizamos o protocolo de três seeds por arquitetura, transferência de aprendizado, métricas preditivas e um benchmark.":
    "Este estudo comparou três arquiteturas da família ResNet — ResNet18, ResNet34 e ResNet50 — na classificação de imagens de satélite por uso e cobertura do solo, utilizando o conjunto EuroSAT RGB. O protocolo incluiu três sementes por arquitetura, aprendizado por transferência, métricas preditivas e um benchmark de eficiência.",
    "Das três arquiteturas a ResNet50 obteve o maior valor de macro-F1 na validação, 98,8644%, porém teve uma maior latência, maior uso de memória e armazenamento. A ResNet18 obteve o maior custo-benefício entre as arquiteturas, o macro-F1 foi de 98,5113% com uma latência bem menor. No teste final obteve os valores de 98,6667% e 98,6122% de acurácia e macro-F1 respectivamente.":
    "Entre as três arquiteturas, a ResNet50 obteve o maior macro-F1 na validação, de 98,8644%, mas apresentou maior latência, uso de memória e demanda de armazenamento. A ResNet18 apresentou a melhor relação custo-benefício, com macro-F1 de 98,5113% e latência significativamente menor. No teste final, obteve acurácia de 98,6667% e macro-F1 de 98,6122%.",
    "Com os experimentos executados, conseguimos alcançar o objetivo principal. Segundo a avaliação, a ResNet18 pré-treinada e ajustada integralmente apresentou o melhor custo-benefício, com uma boa relação entre o desempenho e o custo computacional. Houve uma vantagem preditiva da ResNet50, porém descartável, por conta do aumento de latência, memória e exigência de um espaço maior e armazenamento. Porém tudo depende do objetivo principal, caso cada fração de acurácia seja necessária para uma aplicação crítica por exemplo, a ResNet50 ainda é uma alternativa viável, mas para um processamento mais eficiente a ResNet18 é a recomendação.":
    "Os experimentos permitiram alcançar o objetivo principal. Nas condições avaliadas, a ResNet18 pré-treinada e integralmente ajustada apresentou a melhor relação entre desempenho e custo computacional. A ResNet50 obteve uma pequena vantagem preditiva, acompanhada por aumentos de latência, memória e armazenamento. A escolha, contudo, depende da aplicação: se cada fração de desempenho for necessária em um cenário crítico, a ResNet50 continua sendo uma alternativa viável; para processamento mais eficiente, recomenda-se a ResNet18.",
    "Como um próximo passo, podemos testar imagens provenientes de outras regiões, outras bandas multiespectrais e outras arquiteturas. Também seria interessante investigar a confiança nas previsões, principalmente nas classes agrícolas que apresentaram a maior taxa de erro.":
    "Como trabalhos futuros, podem-se testar imagens de outras regiões, bandas multiespectrais e arquiteturas distintas. Também é relevante investigar a confiança das previsões, principalmente nas classes agrícolas que apresentaram as maiores taxas de erro.",
    "Aluno(a): Eduardo Jeremias dos anjos": "Aluno(a): Eduardo Jeremias dos Anjos",
    "3. Declaração sobre cuidados Éticos e atendimento à LGPD": "3. Declaração sobre cuidados éticos e atendimento à LGPD",
    "Data: 26/09/ 2026": "Data: 26/09/2026",
}


def set_text_preserving_first_run(paragraph, new_text):
    paragraph.clear()
    paragraph.add_run(new_text)


def set_spacing(paragraph, before=0, after=0, line=1.0, keep_next=None):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    if keep_next is not None:
        fmt.keep_with_next = keep_next


def normalized(text):
    return " ".join(text.replace("\u00a0", " ").split())


doc = Document(SOURCE)
counts = {old: 0 for old in REPLACEMENTS}
normalized_replacements = {normalized(old): (old, new) for old, new in REPLACEMENTS.items()}
for paragraph in doc.paragraphs:
    key = normalized(paragraph.text)
    if key in normalized_replacements:
        old, new = normalized_replacements[key]
        set_text_preserving_first_run(paragraph, new)
        counts[old] += 1

# Update the manually composed table of contents with the revised headings and page numbers.
toc_entries = [
    ("1 INTRODUÇÃO", "11", True),
    ("1.1 Problema", "11", False),
    ("1.2 Objetivos", "11", False),
    ("1.3 Organização da pesquisa", "12", False),
    ("2 FUNDAMENTAÇÃO TEÓRICA", "12", True),
    ("2.1 Classificação de imagens e redes convolucionais", "12", False),
    ("2.2 Redes residuais", "13", False),
    ("2.3 ImageNet e aprendizado por transferência", "13", False),
    ("2.4 EuroSAT", "13", False),
    ("2.5 Métricas de desempenho e eficiência", "14", False),
    ("3 METODOLOGIA", "15", True),
    ("3.1 Natureza da pesquisa", "15", False),
    ("3.2 Preparação dos dados", "15", False),
    ("3.3 Configuração de treinamento", "15", False),
    ("3.4 Baseline e ablações", "16", False),
    ("3.5 Benchmark de inferência", "16", False),
    ("3.6 Regra de seleção e uso do teste", "17", False),
    ("4 RESULTADOS E DISCUSSÃO", "17", True),
    ("4.1 Baselines e efeito do aprendizado por transferência", "17", False),
    ("4.2 Comparação entre ResNet18, ResNet34 e ResNet50", "18", False),
    ("4.3 Seleção do modelo", "19", False),
    ("4.4 Avaliação final no teste", "19", False),
    ("4.5 Análise por classe", "21", False),
    ("4.6 Limitações e ameaças à validade", "22", False),
    ("5 CONCLUSÃO", "23", True),
    ("REFERÊNCIAS", "23", True),
    ("APÊNDICE A – DECLARAÇÃO DE USO DE INTELIGÊNCIA ARTIFICIAL", "25", True),
]
toc_started = False
toc_finished = False
for paragraph in doc.paragraphs:
    text = normalized(paragraph.text)
    if text == "SUMÁRIO":
        toc_started = True
        continue
    if not toc_started or toc_finished or not text:
        continue
    if text == "1 INTRODUÇÃO":
        toc_finished = True
        continue
    for title, page, bold in toc_entries:
        # Match the old title before its manually typed dot leaders.
        old_title = text.split(".", 1)[0] if title in {"REFERÊNCIAS"} else None
        if text.startswith(title.split("\t", 1)[0]) or (
            title.startswith("2.3 ") and text.startswith("2.3 ImageNet e transferência de aprendizado")
        ) or (
            title.startswith("3.4 ") and text.startswith("3.4 Baselines")
        ) or (
            title.startswith("3.5 ") and text.startswith("3.5 Benchmark")
        ) or (
            title.startswith("4.1 ") and text.startswith("4.1 Baselines e efeito da transferência de aprendizado")
        ) or (
            title == "APÊNDICE A – DECLARAÇÃO DE USO DE INTELIGÊNCIA ARTIFICIAL" and text.startswith("APÊNDICE A")
        ) or (
            title == "REFERÊNCIAS" and text.lstrip().startswith("REFERÊNCIAS")
        ):
            paragraph.clear()
            run = paragraph.add_run(f"{title}\t{page}")
            run.bold = bold
            run.font.size = Pt(12)
            paragraph.paragraph_format.tab_stops.clear_all()
            paragraph.paragraph_format.tab_stops.add_tab_stop(
                Cm(15.8), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
            )
            break

# Preserve bold labels in the two keyword paragraphs.
for paragraph in doc.paragraphs:
    if paragraph.text.startswith("Palavras-chave:"):
        full = paragraph.text
        paragraph.clear()
        paragraph.add_run("Palavras-chave:").bold = True
        paragraph.add_run(full[len("Palavras-chave:"):])
    elif paragraph.text.startswith("Keywords:"):
        full = paragraph.text
        paragraph.clear()
        paragraph.add_run("Keywords:").bold = True
        paragraph.add_run(full[len("Keywords:"):])

# Put every figure caption above its image and keep caption, image and source together.
paragraphs = list(doc.paragraphs)
for i, paragraph in enumerate(paragraphs[:-2]):
    has_drawing = bool(paragraph._p.xpath(".//w:drawing | .//w:pict"))
    if not has_drawing:
        continue
    caption = paragraphs[i + 1]
    source = paragraphs[i + 2]
    if not caption.text.strip().startswith("Figura "):
        continue
    paragraph._p.addprevious(caption._p)
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    source.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_spacing(caption, before=6, after=3, line=1.0, keep_next=True)
    set_spacing(paragraph, before=0, after=0, line=1.0, keep_next=True)
    set_spacing(source, before=3, after=6, line=1.0, keep_next=False)
    for run in caption.runs:
        run.font.size = Pt(10)
    for run in source.runs:
        run.font.size = Pt(10)

# Normalize all figure caption/source spacing after the reordering pass.
for paragraph in doc.paragraphs:
    stripped = paragraph.text.strip()
    if stripped.startswith("Figura "):
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_spacing(paragraph, before=6, after=3, line=1.0, keep_next=True)
        for run in paragraph.runs:
            run.font.size = Pt(10)
            run.bold = True
    elif stripped.startswith("Fonte:"):
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_spacing(paragraph, before=3, after=6, line=1.0, keep_next=False)
        for run in paragraph.runs:
            run.font.size = Pt(10)

# Prevent headings from being stranded at the bottom of a page.
import re
for paragraph in doc.paragraphs:
    text = paragraph.text.strip()
    if re.match(r"^[1-5](?:\.\d+)?\s+\S", text) or text in {"REFERÊNCIAS", "RESUMO", "ABSTRACT", "SUMÁRIO"}:
        paragraph.paragraph_format.keep_with_next = True

missing = [old for old, count in counts.items() if count == 0]
if missing:
    print("WARNING: replacements not found:")
    for item in missing:
        print(item[:160].encode("unicode_escape").decode("ascii"))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUTPUT)
print(f"saved={OUTPUT}")
print(f"replacements={sum(counts.values())}/{len(REPLACEMENTS)}")
