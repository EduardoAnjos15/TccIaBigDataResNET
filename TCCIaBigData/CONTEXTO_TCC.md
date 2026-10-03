# Contexto do projeto de TCC

Última atualização: 23 de agosto de 2026.

Este documento registra o contexto técnico e metodológico construído durante a conversa sobre o TCC. Ele serve como ponto de retomada caso a conversa seja interrompida ou aberta em outro ambiente.

Regra de continuidade solicitada pelo autor: ao longo desta conversa, atualizar este arquivo sempre que surgir uma nova decisão, execução experimental, resultado, alteração no código, conclusão ou próximo passo relevante. Mensagens sem impacto no projeto podem ser omitidas para que o documento permaneça útil e legível.

## 1. Tema e objetivo

O TCC investiga classificação de imagens de satélite usando redes ResNet implementadas em PyTorch.

Objetivo geral:

> Comparar variantes da arquitetura ResNet para identificar a melhor relação entre desempenho preditivo e custo computacional na classificação de cobertura e uso do solo em imagens de satélite.

Aplicação imaginada: um sistema que recebe uma imagem de satélite de uma região e a classifica em categorias como floresta, área residencial, cultivo, rio ou lago.

Pergunta de pesquisa provisória:

> Qual variante da ResNet oferece a melhor relação entre desempenho de classificação e custo computacional no conjunto EuroSAT?

## 2. Dataset

Foi escolhido o EuroSAT RGB, com 27.000 imagens e 10 classes:

- `AnnualCrop`: cultivo anual, como áreas agrícolas com culturas plantadas e colhidas no mesmo ciclo anual;
- `Forest`: floresta;
- `HerbaceousVegetation`: vegetação herbácea;
- `Highway`: rodovia;
- `Industrial`: área industrial;
- `Pasture`: pastagem;
- `PermanentCrop`: cultivo permanente;
- `Residential`: área residencial;
- `River`: rio;
- `SeaLake`: mar ou lago.

Divisão estratificada oficial:

- treino: 18.900 imagens;
- validação: 4.050 imagens;
- teste: 4.050 imagens.

Arquivo persistente da divisão:

`D:\TCCIaBigData\data\splits\eurosat_stratified_seed42.json`

A divisão é estratificada, mantendo a proporção das classes. Não há sobreposição entre treino, validação e teste.

O conjunto de teste ainda não foi avaliado. Ele deve permanecer intocado até a escolha definitiva do modelo e do protocolo.

## 3. Ambiente computacional

- Sistema operacional: Windows;
- projeto: `D:\TCCIaBigData`;
- ambiente virtual: `D:\TCCIaBigData\.venv`;
- Python: 3.12.9;
- GPU: NVIDIA GeForce RTX 3060 com 12 GB de VRAM;
- PyTorch: 2.12.1+cu126;
- torchvision: 0.27.1+cu126;
- CUDA foi detectado e os treinamentos foram executados com `Dispositivo: cuda`.

Ocorreram problemas iniciais com o Python não encontrado, interferência dos aliases da Microsoft Store, instalação incompleta do PyTorch, arquivo temporário em uso, ambiente Python apontando indevidamente para `D:\Lib` e falta de espaço durante o download do pacote CUDA. Esses problemas foram solucionados. O ambiente atual passou em `pip check` e não possui dependências quebradas.

## 4. Conceitos estabelecidos

### ResNet

ResNet significa Residual Network. Ela utiliza conexões residuais, ou atalhos, que permitem que a informação atravesse blocos da rede. Isso facilita o treinamento de redes profundas e reduz problemas como desaparecimento do gradiente.

ResNet18, ResNet34 e ResNet50 indicam famílias com diferentes profundidades e tipos de blocos. Modelos maiores geralmente possuem mais capacidade, mas também exigem mais tempo, memória e processamento.

### ImageNet

ImageNet é um grande conjunto de imagens naturais usado para pré-treinar modelos. Os pesos aprendidos fornecem características visuais gerais, como bordas, texturas e formas. No projeto, esses pesos são adaptados ao EuroSAT por transferência de aprendizado.

### Baseline

Baseline é uma solução de referência usada para determinar se um modelo mais complexo realmente acrescenta valor.

- Baseline ingênua: sempre prevê a classe mais frequente do treino;
- baseline experimental: ResNet18 pré-treinada, com ajuste completo dos pesos;
- ablações: ResNet18 congelada e ResNet18 treinada do zero.

### Macro-F1

O F1 combina precisão e revocação. O macro-F1 calcula o F1 separadamente para cada classe e depois tira a média simples. Assim, todas as classes têm o mesmo peso, mesmo que possuam quantidades diferentes de imagens.

## 5. Protocolo experimental

Configuração principal:

- arquiteturas: ResNet18, ResNet34 e ResNet50;
- pesos pré-treinados no ImageNet;
- ajuste completo dos pesos;
- 15 épocas;
- batch size 64;
- learning rate 0,0001;
- weight decay 0,0001;
- divisão fixa com seed 42;
- seeds de treinamento: 42, 123 e 2026;
- seleção do checkpoint pela menor perda de validação;
- conjunto de teste reservado para uma única avaliação final.

No código atual:

- `--seed` controla inicialização e aleatoriedade do treinamento;
- `--split-seed` controla a divisão persistente dos dados;
- as repetições usam `--split-seed 42` e variam somente `--seed`.

Essa separação é importante: mudar a divisão juntamente com a seed de treinamento misturaria a variação causada pela inicialização com a variação causada por diferentes imagens nos conjuntos.

Interromper um treinamento com `Ctrl+C` não altera os experimentos concluídos anteriormente. A execução interrompida pode deixar uma pasta parcial; uma nova execução completa em outra pasta ou com substituição controlada produz um resultado válido.

## 6. Arquivos do projeto

- `train.py`: treinamento das ResNets, checkpoint, validação, métricas e curvas;
- `data_utils.py`: download/carregamento do EuroSAT, transforms e divisão estratificada persistente;
- `metrics_utils.py`: acurácia, precisão, recall, F1, macro-F1 e matriz de confusão;
- `naive_baseline.py`: baseline determinística da classe majoritária;
- `evaluate.py`: avaliação final explícita no conjunto de teste;
- `generate_plots.py`: geração de gráficos acadêmicos em PNG 300 dpi e PDF vetorial;
- `generate_dataset_samples.py`: seleção reproduzível e painel de amostras reais do EuroSAT;
- `generate_pareto_plots.py`: consolidação de desempenho e custo, gráficos de Pareto e registro automático da seleção;
- `MODEL_SELECTION.md`: registro prévio e imutável da escolha feita antes do teste;
- `README.md`: instruções de uso;
- `requirements.txt`: dependências;
- `CONTEXTO_TCC.md`: este documento.

Cada treinamento concluído normalmente salva:

- `best_model.pt`: checkpoint selecionado;
- `metrics.json`: configuração, histórico, métricas, tempo e memória;
- `learning_curves.png`: loss e acurácia por época;
- matriz de confusão;
- F1 por classe.

## 7. Baselines e ablações

### Baseline ingênua

Pasta: `outputs\baseline_naive`

A baseline prevê sempre `AnnualCrop`. Há empate entre algumas classes majoritárias e o código escolhe deterministicamente o menor identificador, correspondente a `AnnualCrop`.

- acurácia de validação: 11,11%;
- macro-F1: 2,00%.

### ResNet18 pré-treinada — seed 42

Pasta: `outputs\baseline_resnet18`

- acurácia: 98,4938%;
- macro-F1: 98,4575%;
- melhor época: 15;
- tempo: 1.456,42 s, ou 24,27 min;
- pico de memória: 1.543,84 MB;
- parâmetros totais e treináveis: 11.181.642.

### ResNet18 com backbone congelado — learning rate 0,0001

Pasta: `outputs\resnet18_frozen`

- acurácia: 81,6049%;
- macro-F1: 81,4749%;
- tempo: 16,39 min;
- pico de memória: 531,98 MB;
- parâmetros treináveis: 5.130.

### ResNet18 com backbone congelado — learning rate 0,001

Pasta: `outputs\resnet18_frozen_lr1e3`

- acurácia: 90,2469%;
- macro-F1: 90,0819%;
- melhor época: 14;
- tempo: 12,76 min;
- pico de memória: 531,98 MB.

O learning rate maior foi mais adequado quando somente a camada final estava sendo treinada, mas o ajuste completo ainda foi muito superior.

### ResNet18 treinada do zero

Pasta: `outputs\resnet18_scratch`

- acurácia: 96,2469%;
- macro-F1: 96,1336%;
- melhor época: 15;
- tempo: 18,50 min;
- pico de memória: 1.543,84 MB;
- parâmetros treináveis: 11.181.642.

Sob o orçamento fixo de 15 épocas e learning rate 0,0001, o pré-treinamento no ImageNet aumentou a acurácia em aproximadamente 2,25 pontos percentuais e o macro-F1 em aproximadamente 2,32 pontos. Essa conclusão deve ser apresentada como válida para o orçamento testado; um treinamento do zero mais longo e ajustado poderia produzir outro resultado.

## 8. Comparação principal — resultados individuais

Todos os resultados abaixo são de validação e usam a mesma divisão `seed42`.

| Arquitetura | Seed | Acurácia | Macro-F1 | Tempo de treino | Pico de GPU |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 42 | 98,4938% | 98,4575% | 24,27 min | 1.543,84 MB |
| ResNet18 | 123 | 98,4938% | 98,4570% | 18,48 min | 1.543,84 MB |
| ResNet18 | 2026 | 98,6667% | 98,6194% | 18,47 min | 1.543,84 MB |
| ResNet34 | 42 | 98,5679% | 98,5212% | 25,57 min | 2.282,21 MB |
| ResNet34 | 123 | 98,5432% | 98,5035% | 25,38 min | 2.282,21 MB |
| ResNet34 | 2026 | 98,6420% | 98,6116% | 25,43 min | 2.282,21 MB |
| ResNet50 | 42 | 98,9630% | 98,9337% | 35,64 min | 5.609,92 MB |
| ResNet50 | 123 | 98,8395% | 98,7868% | 36,10 min | 5.609,92 MB |
| ResNet50 | 2026 | 98,9136% | 98,8728% | 35,55 min | 5.609,92 MB |

Observação: o checkpoint não é necessariamente o da época com maior acurácia isolada. O protocolo predefinido seleciona a menor perda de validação.

## 9. Resultados consolidados

Os desvios abaixo são desvios-padrão amostrais entre as três seeds.

| Modelo | Acurácia média | Macro-F1 médio | Tempo médio | Pico de GPU | Parâmetros |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 98,551% ± 0,100 | 98,511% ± 0,094 | 20,41 ± 3,35 min | 1.544 MB | 11,18 milhões |
| ResNet34 | 98,584% ± 0,051 | 98,545% ± 0,058 | 25,46 ± 0,10 min | 2.282 MB | 21,29 milhões |
| ResNet50 | **98,905% ± 0,062** | **98,864% ± 0,074** | 35,76 ± 0,29 min | 5.610 MB | 23,53 milhões |

Comparação média:

- ResNet34 contra ResNet18: aproximadamente +0,033 ponto percentual de acurácia e +0,034 ponto de macro-F1;
- ResNet50 contra ResNet18: aproximadamente +0,354 ponto percentual de acurácia e +0,353 ponto de macro-F1;
- ResNet50 usa aproximadamente 75% mais tempo médio de treinamento, 263% mais memória de GPU e 110% mais parâmetros que a ResNet18.

O tempo da ResNet18 seed 42 foi maior que o das outras duas execuções da mesma arquitetura. Como a arquitetura e a configuração são iguais, essa diferença deve ser tratada como variação do ambiente de execução, cache ou carga do computador. É recomendável realizar um benchmark controlado antes de apresentar conclusões fortes sobre tempo.

## 10. Conclusões provisórias

- Melhor desempenho absoluto: ResNet50;
- melhor custo-benefício provável: ResNet18;
- ResNet34 acrescentou custo considerável e apresentou desempenho praticamente igual ao da ResNet18;
- os resultados das três seeds foram estáveis;
- os modelos superaram amplamente a baseline ingênua;
- transferência de aprendizado foi vantajosa no orçamento experimental adotado;
- congelar todo o backbone reduziu custo, mas causou perda importante de desempenho.

A escolha final ainda não foi formalizada porque falta medir custo de inferência. Para uma aplicação real, custo de treinamento sozinho é insuficiente.

## 11. Gráficos

Gráficos consolidados:

- `outputs\plots\class_distribution.png` e `.pdf`;
- `outputs\plots\baseline_comparison.png` e `.pdf`;
- `outputs\plots\efficiency.png` e `.pdf`.

Cada pasta oficial também contém:

- matriz de confusão normalizada em PNG e PDF;
- F1 por classe em PNG e PDF;
- curva de aprendizado em PNG para modelos treinados.

O script de gráficos agrupa repetições compatíveis, apresenta média e variação quando há mais de uma seed e exclui resultados preliminares incompatíveis com a divisão oficial.

Em 23 de agosto de 2026 foi proposta uma figura qualitativa do dataset com três imagens reais do EuroSAT para cada uma das dez classes, totalizando 30 amostras. A seleção deverá ser automática, reprodutível e feita na partição de treino com seed fixa 42, evitando escolha manual ou favorecimento de exemplos visualmente fáceis. A figura deve mostrar o nome original e, se possível, a tradução de cada classe, ser exportada em PNG 300 dpi e PDF e citar o EuroSAT como fonte. Uma segunda figura, produzida posteriormente, poderá apresentar erros e confusões representativos do modelo escolhido usando validação, sem acessar antecipadamente o teste.

Implementação preparada: `generate_dataset_samples.py`. O script seleciona exemplos apenas de treino ou validação, usa uma seed de seleção independente, exporta as 30 imagens individuais sem transformações, cria um painel acadêmico em PNG 300 dpi e PDF e salva um manifesto JSON com os índices e caminhos. A opção de teste foi deliberadamente excluída. O benchmark de inferência não será usado para escolher essas amostras representativas; ele mede custo computacional. A inferência na validação poderá ser utilizada mais tarde, em uma análise separada, para encontrar erros e casos difíceis do modelo escolhido.

O script foi executado com partição de treino, três amostras por classe, split seed 42 e selection seed 42. A validação confirmou 30 índices únicos, todos pertencentes ao treino e nenhum pertencente ao teste. Foram gerados `outputs\dataset_samples\eurosat_train_samples.png`, a versão PDF, o manifesto `eurosat_train_samples_manifest.json` e 30 PNGs individuais organizados por classe em `outputs\dataset_samples\selected_images`. O painel foi inspecionado visualmente e apresenta duas classes por linha, com três amostras e rótulos em português e inglês para cada classe.

Em seguida, o gerador foi ampliado para produzir também dez painéis individuais em `outputs\dataset_samples\class_panels`. Cada painel apresenta a identificação da classe em português e inglês no topo e os três exemplos alinhados abaixo, além de versões PNG 300 dpi e PDF. Esses arquivos são mais adequados para inserir separadamente no texto do TCC.

A execução e a inspeção visual foram concluídas: existem dez PNGs e dez PDFs, um par para cada classe, e o manifesto registra os caminhos de todos eles. O painel de cultivo anual foi usado na verificação visual e confirmou legenda legível, três imagens alinhadas e ausência de cortes ou sobreposições.

Resultados antigos como `outputs\e1_resnet18_pretrained`, `outputs\run1` e `outputs\smoke_test` foram preservados, mas não fazem parte da comparação oficial.

## 12. Estado dos dados salvos

Uma auditoria confirmou:

- 16 arquivos de métricas JSON válidos;
- nenhum JSON corrompido;
- checkpoints presentes em todos os treinamentos concluídos;
- aproximadamente 911,4 MB de resultados após a última repetição da ResNet50;
- a presença de um arquivo chamado `outputs\curve_test.png`, que é apenas um gráfico antigo e não representa avaliação do conjunto de teste.

Todos os dados estão salvos localmente no disco `D:`. Isso não substitui backup. Recomenda-se copiar ao menos os códigos, `data\splits`, `outputs` e este documento para outro disco ou armazenamento em nuvem.

## 13. Próximos passos

Ordem recomendada:

1. Criar um benchmark de inferência usando somente a validação;
2. medir latência por imagem, throughput em imagens por segundo, memória de inferência e tamanho do checkpoint;
3. repetir o benchmark algumas vezes com aquecimento da GPU;
4. consolidar desempenho e custo em uma tabela e um gráfico de Pareto;
5. definir formalmente o modelo vencedor antes de acessar o teste;
6. avaliar uma única vez no teste o modelo e checkpoint escolhidos;
7. gerar matriz de confusão, métricas por classe e macro-F1 do teste;
8. documentar metodologia, resultados, limitações e ameaças à validade no texto do TCC.

Próximo passo imediato definido em 23 de agosto de 2026: implementar e executar um benchmark de inferência na partição de validação para ResNet18, ResNet34 e ResNet50. O benchmark deve usar os checkpoints da seed 42 como representantes de cada arquitetura, pois os pesos não alteram materialmente o custo estrutural e essa regra uniforme evita escolher uma seed favorável. Devem ser medidos, com aquecimento da GPU e sincronização CUDA, latência por imagem, throughput, pico de memória de inferência e tamanho do checkpoint. Recomenda-se medir batch size 1, representando uma aplicação interativa, e batch size 64, representando processamento em lote, com múltiplas repetições para obter média e desvio-padrão. O conjunto de teste continuará intocado durante essa etapa.

Estado corrente: a seleção e a apresentação qualitativa das imagens do EuroSAT foram concluídas. A próxima atividade prática é criar `benchmark_inference.py`, validar sua metodologia e executar as três arquiteturas nos checkpoints da seed 42, usando somente a partição de validação.

### Benchmark de inferência concluído

O arquivo `benchmark_inference.py` foi implementado, validado com execuções curtas e executado oficialmente. Resultados: `outputs\inference_benchmark\benchmark_results.json` e `benchmark_summary.csv`. Os diretórios `inference_benchmark_smoke` e `inference_benchmark_memory_check` contêm apenas verificações técnicas e não devem ser tratados como resultados oficiais.

Protocolo oficial de inferência:

- checkpoints da seed de treinamento 42;
- partição de validação com 4.050 imagens e split seed 42;
- batch sizes 1 e 64;
- cinco repetições do pipeline completo;
- 20 lotes de aquecimento;
- 100 forwards sincronizados por repetição na medição exclusiva do modelo;
- precisão FP32, GPU RTX 3060 e modo `torch.inference_mode()`;
- tempo do forward medido com eventos CUDA;
- pipeline medido por tempo de parede, incluindo leitura, transformação, transferência e modelo;
- conjunto de teste não acessado.

Resultados oficiais com batch 1:

| Modelo | Latência do modelo (ms/imagem) | Throughput do modelo (img/s) | Throughput do pipeline (img/s) | Pico sincronizado (MB) | Checkpoint (MB) |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 2,402 ± 0,045 | 416,37 ± 7,79 | 232,68 ± 0,52 | 60,98 | 42,73 |
| ResNet34 | 4,478 ± 0,148 | 223,49 ± 7,12 | 156,49 ± 0,49 | 99,58 | 81,35 |
| ResNet50 | 6,152 ± 0,079 | 162,57 ± 2,08 | 120,54 ± 0,11 | 110,82 | 90,06 |

Resultados oficiais com batch 64:

| Modelo | Latência do modelo (ms/imagem) | Throughput do modelo (img/s) | Throughput do pipeline (img/s) | Pico sincronizado (MB) | Checkpoint (MB) |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 0,516 ± 0,001 | 1.938,76 ± 3,86 | 749,18 ± 12,85 | 479,58 | 42,73 |
| ResNet34 | 0,918 ± 0,002 | 1.089,18 ± 2,74 | 729,57 ± 16,58 | 518,18 | 81,35 |
| ResNet50 | 1,601 ± 0,0001 | 624,58 ± 0,04 | 586,52 ± 24,58 | 773,24 | 90,06 |

As acurácias observadas no pipeline coincidiram com os checkpoints de validação: 98,4938% para ResNet18, 98,5679% para ResNet34 e 98,9630% para ResNet50.

O JSON também registra o pico do pipeline assíncrono completo. Em batch 64 ele foi 1.269,59 MB para ResNet18, 554,93 MB para ResNet34 e 2.559,69 MB para ResNet50. Esse valor pode variar com enfileiramento assíncrono, cache e escolha de algoritmos do cuDNN. Para uma comparação estrutural mais justa entre arquiteturas, a tabela principal usa o pico do forward sincronizado; o pico do pipeline deve ser relatado separadamente com essa ressalva.

Interpretação preliminar: com batch 1, a ResNet18 foi aproximadamente 2,56 vezes mais rápida por imagem que a ResNet50. Com batch 64, foi aproximadamente 3,10 vezes mais rápida no forward. A ResNet50 manteve o melhor desempenho preditivo, mas seu ganho médio sobre a ResNet18 foi de aproximadamente 0,354 ponto percentual de acurácia. Os resultados fortalecem a ResNet18 como provável vencedora em custo-benefício e a ResNet50 como vencedora em desempenho absoluto. A decisão formal ainda deve ser registrada após a geração do gráfico de Pareto e antes da avaliação final no teste.

Próxima etapa corrente: criar gráficos acadêmicos que combinem a média e o desvio-padrão do macro-F1 das três seeds com as métricas oficiais de inferência. Devem ser produzidas relações entre macro-F1 e latência com batch 1, macro-F1 e throughput com batch 64, macro-F1 e pico de memória sincronizado e macro-F1 e tamanho do checkpoint. Depois dos gráficos, registrar uma regra de decisão e o modelo selecionado, incluindo o checkpoint que será usado, sem consultar o teste. Somente após esse registro será autorizada a avaliação final.

### Gráficos de Pareto e seleção formal concluídos

O script `generate_pareto_plots.py` foi implementado e executado. Ele consolidou exclusivamente as execuções oficiais das seeds 42, 123 e 2026, confirmou o mesmo split do benchmark completo e verificou que ainda não existia qualquer `test_metrics.json`.

Foram gerados, em PNG 300 dpi e PDF vetorial:

- `outputs\plots\pareto_latency_batch1`;
- `outputs\plots\pareto_throughput_batch64`;
- `outputs\plots\pareto_memory_batch64`;
- `outputs\plots\pareto_checkpoint_size`;
- `outputs\plots\pareto_overview`.

A tabela-fonte está em `outputs\plots\pareto_summary.csv`. Os gráficos usam o macro-F1 médio de três seeds, barras com desvio-padrão amostral, custo de inferência oficial e uma estrela para o modelo selecionado. O painel consolidado foi inspecionado visualmente e não apresentou cortes, sobreposições ou rótulos ilegíveis.

Regra registrada antes do teste:

1. considerar elegíveis as arquiteturas até 0,5 ponto percentual abaixo da melhor média de macro-F1 na validação;
2. entre as elegíveis, selecionar a menor latência média do modelo com batch 1;
3. utilizar o checkpoint da seed 42, regra uniforme definida antes do benchmark.

As três arquiteturas ficaram dentro da margem de desempenho. A aplicação da regra selecionou formalmente a **ResNet18 pré-treinada com ajuste completo**, checkpoint `D:\TCCIaBigData\outputs\baseline_resnet18\best_model.pt`. Ela obteve macro-F1 médio de 98,5113% ± 0,0936% e latência de 2,402 ± 0,045 ms/imagem com batch 1. A ResNet50 apresentou macro-F1 médio 0,353 ponto percentual maior, mas latência 2,56 vezes maior e checkpoint 2,11 vezes maior.

O registro oficial está em `MODEL_SELECTION.md` e `outputs\model_selection\selection_decision.json`. No momento do registro, o conjunto de teste não havia sido avaliado. A escolha não deverá ser alterada em função do resultado final. Próxima etapa autorizada: executar uma única avaliação do checkpoint selecionado na partição de teste e salvar suas métricas finais.

Comando definido para a próxima execução:

```powershell
python evaluate.py `
  --checkpoint outputs/baseline_resnet18/best_model.pt `
  --batch-size 64 `
  --no-download `
  --output-dir outputs/final_resnet18_test
```

Após essa execução, o resultado do teste será tratado apenas como estimativa final de generalização. Ele não poderá motivar troca de arquitetura, checkpoint, seed ou hiperparâmetro. Restará gerar os gráficos finais do teste, analisar os erros por classe e redigir a conclusão experimental.

Para o benchmark de inferência, devem ser comparados checkpoints equivalentes. Uma opção metodológica simples é usar a seed 42 de cada arquitetura, ou escolher de antemão uma regra uniforme para selecionar uma das três seeds. Não se deve escolher a seed com base no conjunto de teste.

### Avaliação final no teste concluída

Em 23 de agosto de 2026, o checkpoint previamente selecionado da ResNet18 foi avaliado uma única vez nas 4.050 imagens da partição de teste, com o comando registrado antes da abertura do teste. O resultado foi:

- acurácia de **98,6667%**;
- macro-F1 de **98,6122%**;
- tempo total de inferência registrado de aproximadamente 74,68 segundos;
- diferença em relação à validação do mesmo checkpoint de +0,173 ponto percentual em acurácia e +0,155 ponto percentual em macro-F1.

O desempenho de teste ficou muito próximo e ligeiramente acima do desempenho de validação. Isso indica boa generalização dentro do EuroSAT e ausência de queda relevante, sem justificar qualquer alteração posterior da arquitetura ou dos hiperparâmetros.

O arquivo principal da avaliação é `outputs\final_resnet18_test\test_metrics.json`. A matriz de confusão original está em `test_confusion_matrix.png`.

O script `generate_final_test_report.py` foi criado e executado somente sobre os JSONs já salvos, sem realizar nova inferência. Antes de gerar os artefatos, ele confirma que arquitetura, checkpoint e divisão correspondem à decisão formal em `outputs\model_selection\selection_decision.json`. Foram produzidos:

- `outputs\final_resnet18_test\final_test_report.md`;
- `final_test_per_class.csv`;
- `top_confusions.json`;
- `test_confusion_matrix_normalized.png` e `.pdf`;
- `test_f1_per_class.png` e `.pdf`;
- `validation_vs_test.png` e `.pdf`;
- `validation_vs_test_f1_per_class.png` e `.pdf`.

Os gráficos foram inspecionados visualmente e estão legíveis. A classe de menor F1 no teste foi cultivo permanente, com 97,30%. As principais confusões dirigidas foram cultivo permanente classificado como cultivo anual, com 8 imagens, e como vegetação herbácea, com 6 imagens. Floresta e mar/lago apresentaram os melhores resultados, com F1 de 99,67% e 99,78%, respectivamente. Esse padrão é plausível porque as categorias agrícolas compartilham texturas, cores e padrões espaciais semelhantes nas imagens de satélite.

Estado corrente: o protocolo experimental principal está concluído — comparação na validação, benchmark de custo, seleção formal anterior ao teste e avaliação final única. O próximo passo da pesquisa é consolidar tabelas e figuras no texto do TCC e redigir metodologia, resultados, discussão, limitações e conclusão, deixando claro que os resultados demonstram generalização no EuroSAT, mas não garantem o mesmo desempenho em imagens de outros satélites, regiões, resoluções ou épocas do ano.

## 14. Comando-base para treinamentos repetidos

Exemplo:

```powershell
python train.py `
  --architecture resnet18 `
  --epochs 15 `
  --batch-size 64 `
  --learning-rate 0.0001 `
  --seed 123 `
  --split-seed 42 `
  --no-download `
  --output-dir outputs/resnet18_pretrained_seed123
```

## 15. Referência discutida

Foi discutida uma postagem do Medium sobre classificação de imagens com ResNet e PyTorch. Referência limpa, sem dados de autenticação:

<https://medium.com/@anglilian/image-classification-with-resnet-pytorch-1e48a4c33905>

A postagem foi usada como inspiração e referência de práticas, não como texto a ser copiado. O trabalho permanece original quando código, redação, experimentos, análise e decisões são próprios e todas as fontes externas utilizadas são citadas adequadamente.

## 16. Rascunho do TCC no modelo Word

Em 23 de agosto de 2026 foi criado um novo documento Word a partir do modelo institucional, preservando o arquivo original. O novo arquivo é:

- `D:\TCCIaBigData\TCC_ResNet_Rascunho_Experimental.docx`.

O modelo original permanece em `D:\TCCIaBigData\TCC_ResNet_Completo_Template.docx`, com SHA-256 `ABAE7A032342053593F46E2E3C662E32AEE6DACECE8496D3FD6A0D37D2CF4A98`, igual ao valor conferido antes da criação do rascunho.

O rascunho possui 24 páginas e inclui elementos pré-textuais, resumo, abstract, sumário, introdução, fundamentação teórica, metodologia, resultados e discussão, conclusão e referências. Foram incorporadas cinco tabelas e seis figuras com os resultados reais do projeto: exemplos do EuroSAT, curvas de aprendizado, relações de Pareto, comparação entre validação e teste, matriz de confusão normalizada e F1 por classe.

O documento registra o protocolo experimental completo, as baselines e ablações, as três seeds por arquitetura, o benchmark de inferência, a regra de seleção anterior ao teste e a avaliação final única da ResNet18. A redação também explicita as limitações de generalização para outros sensores, regiões, resoluções e condições temporais.

A versão final foi renderizada em 24 páginas e inspecionada visualmente página por página. Não foram observados cortes, sobreposições, tabelas truncadas ou campos pendentes. A integridade interna do pacote DOCX foi validada, e não foram encontrados marcadores como `a definir`, `xxxx`, nomes fictícios ou instruções do modelo. Dedicatória, agradecimentos e demais textos pessoais devem ser revisados pelo autor antes da entrega oficial.

O gerador reproduzível do rascunho está em `D:\TCCIaBigData\build_tcc_draft.py`. Para recriar a versão com a paginação conferida, utilizar `--page-count 24`, sempre com um caminho de saída diferente do modelo original.

## 17. Síntese explicativa da pesquisa

A pesquisa investiga qual arquitetura da família ResNet oferece o melhor equilíbrio entre qualidade de classificação e custo computacional para reconhecer tipos de uso e cobertura do solo em imagens de satélite. O conjunto EuroSAT RGB fornece 27.000 imagens distribuídas em dez classes, como floresta, cultivo anual, cultivo permanente, área residencial, área industrial, rodovia, rio e mar/lago.

Foram comparadas ResNet18, ResNet34 e ResNet50. O número identifica aproximadamente a profundidade da rede: modelos mais profundos possuem mais camadas e maior capacidade, mas normalmente consomem mais memória, armazenamento e tempo de processamento. Os modelos principais foram inicializados com pesos aprendidos no ImageNet e depois ajustados ao EuroSAT, procedimento chamado transferência de aprendizado.

O conjunto foi separado uma única vez, de forma estratificada, em 70% para treino, 15% para validação e 15% para teste. Treino é a fase em que o modelo aprende; validação é usada para comparar configurações e selecionar o modelo; teste é reservado para a avaliação final. A divisão permaneceu fixa com seed 42, e cada arquitetura foi treinada com as seeds 42, 123 e 2026 para medir a variação causada pela aleatoriedade.

A baseline ingênua, que sempre prevê a classe mais frequente, estabeleceu um limite inferior de 11,11% de acurácia e 2,00% de macro-F1. Também foram avaliadas uma ResNet18 treinada do zero e versões com o extrator de características congelado. Esses experimentos demonstraram que a transferência de aprendizado e o ajuste completo dos pesos melhoraram substancialmente o desempenho.

As principais métricas preditivas foram acurácia e macro-F1. Acurácia é a proporção total de previsões corretas. F1 combina precisão e recall; macro-F1 calcula o F1 de cada classe e depois obtém uma média sem favorecer classes maiores. Para avaliar eficiência, foram medidos latência, throughput, memória e tamanho do checkpoint. Latência é o tempo necessário por previsão; throughput é a quantidade de imagens processadas por segundo; checkpoint é o arquivo que armazena os pesos aprendidos.

Na validação, a ResNet50 alcançou o melhor macro-F1 médio, 98,8644%. A ResNet18 atingiu 98,5113%, diferença aproximada de 0,353 ponto percentual, mas apresentou latência cerca de 2,56 vezes menor em inferência unitária e checkpoint aproximadamente 2,11 vezes menor. A ResNet34 ficou entre as duas, sem oferecer ganho suficiente para compensar seu custo adicional.

Uma regra definida antes da abertura do teste considerou elegíveis os modelos situados até 0,5 ponto percentual do melhor macro-F1 e, entre eles, escolheu a menor latência com batch 1. Por essa regra, a ResNet18 foi selecionada como melhor custo-benefício. No teste final único, com 4.050 imagens, alcançou 98,6667% de acurácia e 98,6122% de macro-F1.

A conclusão é que a ResNet50 possui o melhor desempenho absoluto no cenário estudado, mas a ResNet18 oferece o melhor compromisso entre desempenho e eficiência. Isso a torna a candidata mais adequada para uma futura aplicação de classificação de terrenos por imagens de satélite quando velocidade, memória e tamanho do modelo também forem importantes. Os resultados são válidos para o EuroSAT RGB e não garantem o mesmo desempenho em outros satélites, regiões, resoluções ou épocas do ano.

## 18. Obtenção e preparação dos dados EuroSAT

O projeto não capturou imagens diretamente de um satélite nem realizou rotulagem manual. Foi utilizado o conjunto público EuroSAT, criado por Patrick Helber, Benjamin Bischke, Andreas Dengel e Damian Borth a partir de imagens da missão Sentinel-2 do programa europeu Copernicus. O conjunto completo possui 27.000 recortes de imagem georreferenciados, distribuídos em dez classes de uso e cobertura do solo. A publicação oficial oferece uma versão multiespectral com as 13 bandas do Sentinel-2 e uma versão RGB em JPEG. Este projeto utiliza exclusivamente a versão RGB, formada pelas bandas visíveis vermelha, verde e azul.

No código, a obtenção ocorreu automaticamente por meio de `torchvision.datasets.EuroSAT`. Na primeira execução de `train.py` sem a opção `--no-download`, a instrução `datasets.EuroSAT(root=args.data_dir, download=True)` baixou um arquivo compactado por um espelho do EuroSAT mantido no Hugging Face pelo projeto TorchGeo. A URL está fixada na implementação instalada do `torchvision`, e o arquivo é verificado por MD5 para detectar download corrompido. A cópia presente no projeto é `data\eurosat\EuroSAT.zip`, possui 94.280.567 bytes e MD5 `c8fa014336c82ac7804f0398fcb19387`. Após a verificação, o `torchvision` extraiu as imagens para `data\eurosat\2750`, mantendo uma pasta para cada classe.

A distribuição original carregada pelo projeto é: 3.000 imagens de cultivo anual, 3.000 de floresta, 3.000 de vegetação herbácea, 2.500 de rodovia, 2.500 de área industrial, 2.000 de pastagem, 2.500 de cultivo permanente, 3.000 de área residencial, 2.500 de rio e 3.000 de mar/lago. Os nomes das pastas funcionam como rótulos, e o `ImageFolder` usado internamente pelo `torchvision` associa cada arquivo ao índice de sua classe.

Depois do carregamento, `data_utils.py` realizou uma divisão estratificada: os exemplos de cada classe foram embaralhados separadamente com seed 42 e repartidos em 70% para treino, 15% para validação e 15% para teste. Isso produziu 18.900, 4.050 e 4.050 imagens, respectivamente. Os índices foram congelados em `data\splits\eurosat_stratified_seed42.json`, garantindo que todos os modelos fossem treinados e comparados com exatamente as mesmas imagens em cada partição e sem duplicação entre elas.

No treino, as imagens receberam recorte redimensionado aleatório, espelhamentos horizontal e vertical, rotação de até 20 graus, conversão para tensor e normalização do ImageNet. Na validação e no teste não houve aumento aleatório: foram usados redimensionamento, recorte central, conversão para tensor e a mesma normalização. Portanto, as imagens originais foram obtidas do EuroSAT público, enquanto divisão e pré-processamento foram definidos pelo próprio projeto para assegurar comparação justa e reproduzível.

Formulação curta para apresentação: “O EuroSAT é um conjunto público de classificação de uso e cobertura do solo criado a partir de imagens do Sentinel-2. Utilizamos sua versão RGB com 27.000 imagens e dez classes. Os dados foram baixados automaticamente pelo `torchvision`, verificados por hash, organizados por classe e depois divididos de forma estratificada e reproduzível em 70% para treino, 15% para validação e 15% para teste. Portanto, não coletamos nem rotulamos as imagens; nossa contribuição está no protocolo experimental, na comparação das ResNets e na análise de custo computacional.”

## 19. Uso de inteligência artificial e integridade acadêmica

Existem ferramentas que estimam se um texto pode ter sido gerado por inteligência artificial, como o indicador de escrita por IA do Turnitin e serviços independentes. Esses sistemas não comprovam autoria: podem marcar textos humanos como IA e deixar de identificar textos gerados por IA. O próprio Turnitin informa que falsos positivos são possíveis e trata resultados inferiores a 20% como menos confiáveis; a UNESCO também registra que há pouca evidência de eficácia suficiente para usar detectores como prova isolada.

Neste TCC, a proteção acadêmica adequada não consiste em tentar contornar detectores. O autor deve conhecer e cumprir a política da instituição e do orientador, declarar o uso de IA caso isso seja exigido ou permitido, revisar criticamente toda a redação, verificar as referências e escrever a versão final com sua própria argumentação. Os resultados experimentais possuem rastreabilidade própria no projeto: código-fonte, arquivo de divisão, configurações, checkpoints, métricas, gráficos e registros da seleção do modelo. Esses materiais, junto ao histórico de versões e à capacidade de explicar e defender as decisões metodológicas, constituem evidência mais significativa de autoria e domínio do que a pontuação de um detector.

Antes da entrega, recomenda-se: confirmar as regras institucionais sobre IA; revisar o texto parágrafo por parágrafo; remover afirmações sem fonte; conferir todas as citações; preservar rascunhos e histórico de alterações; e preparar-se para explicar o código, as métricas, os resultados e as limitações. Se a instituição permitir IA como apoio, pode ser incluída uma declaração transparente informando que a ferramenta auxiliou na organização, revisão linguística e suporte à implementação, enquanto o autor definiu o problema, executou os experimentos, verificou os resultados e assume responsabilidade integral pelo conteúdo final.

Reescrever integralmente o rascunho com as próprias palavras é uma prática recomendada, desde que a reescrita seja feita a partir da compreensão do conteúdo e das fontes, e não apenas pela troca mecânica de sinônimos. O autor deve conseguir justificar cada afirmação, explicar as escolhas metodológicas e interpretar os resultados apresentados. As ideias, definições, dados e métodos provenientes de terceiros continuam exigindo citação mesmo quando estão parafraseados. A reescrita humana reduz a dependência textual da IA, mas não elimina eventual obrigação institucional de declarar que ela foi utilizada no planejamento, no código ou na elaboração do rascunho inicial.

## 20. Sugestões de título

O título recomendado para representar com maior precisão o trabalho é: **Comparação de arquiteturas ResNet para classificação de uso e cobertura do solo: desempenho preditivo e custo computacional no EuroSAT**. Ele explicita a natureza comparativa da pesquisa, a família de modelos, a tarefa investigada, os dois eixos de avaliação e o conjunto de dados.

Outras opções adequadas são:

- **Desempenho preditivo e eficiência computacional de arquiteturas ResNet na classificação do EuroSAT**;
- **Análise de custo-benefício de redes ResNet para classificação de uso e cobertura do solo em imagens de satélite**;
- **Comparação entre ResNet18, ResNet34 e ResNet50 na classificação de imagens de satélite**;
- **Classificação de uso e cobertura do solo com ResNet: um estudo experimental no EuroSAT**;
- **Transferência de aprendizado com ResNet para classificação do EuroSAT: desempenho e custo de inferência**;
- **Seleção de uma arquitetura ResNet eficiente para classificação de uso e cobertura do solo**.

Para linguagem acadêmica, recomenda-se preferir “uso e cobertura do solo” a “tipos de terreno”. A expressão “custo computacional” é mais precisa que “custo-benefício” quando são avaliados latência, throughput, memória e tamanho do checkpoint, pois não houve mensuração financeira. O título do documento Word ainda não foi alterado; qualquer mudança deverá ser aplicada posteriormente ao novo rascunho, preservando o modelo original.

## 21. Significado de “24 f.” na referência do trabalho

Na referência bibliográfica do próprio TCC, `24 f.` significa **24 folhas**; `f.` é a abreviação de folha. O número foi preenchido porque o rascunho Word renderizado possuía 24 páginas/folhas naquele momento. Esse valor não é definitivo: deve ser atualizado somente depois da revisão e paginação finais do documento. “Folha” designa a lâmina física, enquanto “página” pode designar cada face da folha. Em trabalhos acadêmicos brasileiros é comum o modelo institucional solicitar a quantidade seguida de `f.`; deve-se manter a convenção indicada pelo modelo da instituição. O número não representa capítulos, figuras, referências ou quantidade de arquivos.

## 22. Resumo completo da pesquisa

A pesquisa teve como objetivo comparar ResNet18, ResNet34 e ResNet50 para descobrir qual arquitetura oferece o melhor compromisso entre qualidade de classificação e custo computacional na identificação de uso e cobertura do solo em imagens de satélite. A contribuição não é a criação de uma nova arquitetura ou de um novo conjunto de dados, mas a realização de um protocolo comparativo controlado, reproduzível e orientado a uma futura aplicação.

Foi utilizada a versão RGB pública do EuroSAT, derivada de imagens Sentinel-2. O conjunto possui 27.000 imagens em dez classes: cultivo anual, floresta, vegetação herbácea, rodovia, área industrial, pastagem, cultivo permanente, área residencial, rio e mar/lago. O download ocorreu pelo `torchvision.datasets.EuroSAT`, e as imagens foram divididas de forma estratificada e fixa, com seed 42, em 18.900 exemplos de treino, 4.050 de validação e 4.050 de teste. Os índices foram salvos em `data\splits\eurosat_stratified_seed42.json`. Assim, todas as arquiteturas utilizaram exatamente os mesmos exemplos em cada partição.

No treino foram empregados recorte redimensionado aleatório para 224 × 224 pixels, espelhamentos horizontal e vertical, rotação de até 20 graus, conversão para tensor e normalização do ImageNet. Validação e teste utilizaram redimensionamento, recorte central de 224 × 224 e a mesma normalização, sem aumentos aleatórios. Os experimentos foram executados em PyTorch com CUDA. A camada classificadora de cada ResNet foi substituída por uma camada com dez saídas.

O protocolo principal usou pesos pré-treinados no ImageNet e ajuste completo de todos os pesos, 15 épocas, batch size 64, learning rate 0,0001, weight decay 0,0001, perda cross-entropy, otimizador AdamW, `ReduceLROnPlateau` e early stopping com paciência de cinco épocas. O melhor checkpoint de cada execução foi definido pela menor perda de validação. Para medir a variação aleatória, cada arquitetura foi treinada com as seeds 42, 123 e 2026, mantendo a divisão dos dados fixada pela split-seed 42.

A baseline ingênua sempre previu uma das classes mais frequentes do treino, cultivo anual, e atingiu 11,11% de acurácia e 2,00% de macro-F1 na validação. As ablações com ResNet18 mostraram: backbone congelado e learning rate 0,0001, 81,60% de acurácia e 81,47% de macro-F1; backbone congelado e learning rate 0,001, 90,25% e 90,08%; treinamento do zero, 96,25% e 96,13%; transferência de aprendizado com ajuste completo na seed 42, 98,49% e 98,46%. Esses resultados evidenciaram a importância do pré-treinamento e mostraram que apenas treinar a camada final foi inferior ao ajuste completo.

Nas nove execuções principais, as médias de validação foram: ResNet18, 98,5514% de acurácia e 98,5113% de macro-F1; ResNet34, 98,5844% e 98,5454%; ResNet50, 98,9053% e 98,8644%. Os desvios-padrão do macro-F1 foram aproximadamente 0,0936, 0,0580 e 0,0738 ponto percentual, respectivamente. A ResNet50 venceu em desempenho absoluto, mas sua vantagem média sobre a ResNet18 foi de apenas 0,353 ponto percentual de macro-F1. A ResNet34 apresentou desempenho praticamente igual ao da ResNet18, apesar do custo maior.

O benchmark de inferência foi executado sobre a validação com os checkpoints da seed 42. Em batch 1, as latências exclusivas do modelo foram 2,402 ms/imagem para ResNet18, 4,478 para ResNet34 e 6,152 para ResNet50. Em batch 64, os throughputs foram 1.938,76, 1.089,18 e 624,58 imagens por segundo. Os picos de memória do forward com batch 64 foram 479,58 MB, 518,18 MB e 773,24 MB. Os checkpoints ocuparam 42,73 MB, 81,35 MB e 90,06 MB, e os modelos possuíam aproximadamente 11,18 milhões, 21,29 milhões e 23,53 milhões de parâmetros.

Antes de acessar o teste, foi formalizada a seguinte regra: considerar elegíveis as arquiteturas situadas até 0,5 ponto percentual abaixo do melhor macro-F1 médio e, entre elas, escolher a de menor latência com batch 1. As três arquiteturas ficaram dentro da margem. A ResNet18 foi selecionada por ser a mais rápida. Em comparação com ela, a ResNet50 apresentou latência 2,56 vezes maior e checkpoint 2,11 vezes maior para obter apenas 0,353 ponto percentual adicional de macro-F1 na validação.

Somente depois dessa decisão, o checkpoint `outputs\baseline_resnet18\best_model.pt` foi avaliado uma única vez nas 4.050 imagens de teste. A ResNet18 alcançou 98,6667% de acurácia e 98,6122% de macro-F1. O teste ficou próximo e ligeiramente acima da validação do mesmo checkpoint, sem queda relevante. Os maiores F1 foram obtidos por mar/lago, 99,78%, e floresta, 99,67%. O menor foi cultivo permanente, 97,30%. As principais confusões foram cultivo permanente classificado como cultivo anual, oito imagens, e como vegetação herbácea, seis imagens, comportamento plausível devido à semelhança visual entre classes agrícolas.

Foram produzidos checkpoints, arquivos JSON com configurações e métricas, curvas de aprendizado, matrizes de confusão, gráficos de F1 por classe, distribuição das classes, comparações entre modelos, gráficos de Pareto, benchmark de inferência, painéis com três exemplos de cada classe, relatório final e um rascunho Word separado do modelo institucional original. Os artefatos permitem rastrear os experimentos e reproduzir as principais decisões.

A conclusão é dupla: a ResNet50 é a melhor quando se considera somente desempenho preditivo; a ResNet18 é a melhor escolha quando desempenho e eficiência são analisados em conjunto. A ResNet18 manteve macro-F1 superior a 98,6% no teste, com menor latência, maior throughput e arquivo menor, sendo a candidata mais adequada para uma futura aplicação com restrições computacionais.

As conclusões estão limitadas ao EuroSAT RGB e ao ambiente avaliado. O estudo não testou outras famílias de redes, outros sensores, outras regiões, diferentes épocas do ano, bandas multiespectrais ou validação externa. A divisão foi estratificada por imagem, não por região geográfica. Também não foram otimizados hiperparâmetros específicos para cada arquitetura. O modelo realiza classificação de um recorte inteiro, e não segmentação pixel a pixel. Portanto, o resultado demonstra generalização dentro do EuroSAT, mas não garante o mesmo desempenho em imagens reais de outra origem.

O protocolo experimental principal está encerrado. A etapa restante é acadêmico-editorial: reescrever o texto com autoria própria, conferir e completar as referências, ajustar título e dados pessoais, atualizar a quantidade final de folhas, selecionar as tabelas e figuras mais relevantes e preparar a explicação oral para a banca. Novos experimentos após a abertura do teste devem ser apresentados como extensões separadas, sem alterar retrospectivamente a seleção já registrada.

## 23. Significado de pesos pré-treinados

Os pesos são os valores numéricos internos que a rede neural ajusta durante o treinamento para reconhecer padrões nas imagens. Pesos pré-treinados são valores obtidos anteriormente ao treinar a mesma arquitetura em outro conjunto de dados. Neste projeto, as ResNets principais começaram com pesos aprendidos no ImageNet, em vez de começarem com valores aleatórios. Esses pesos já codificavam detectores úteis de bordas, cores, texturas, formas e outras características visuais. A camada classificadora original foi substituída por uma nova camada com dez saídas, e a rede foi então adaptada ao EuroSAT.

Quando todos os pesos pré-treinados foram atualizados com o EuroSAT, ocorreu ajuste completo ou fine-tuning. Quando o backbone foi congelado, somente a nova camada final aprendeu. No experimento `scratch`, a ResNet começou sem pesos pré-treinados e precisou aprender apenas com o EuroSAT. A comparação mostrou macro-F1 de 98,46% para a ResNet18 pré-treinada com ajuste completo, 96,13% para treinamento do zero e entre 81,47% e 90,08% para as versões com backbone congelado. Portanto, os pesos pré-treinados ofereceram um ponto de partida mais informativo, mas o melhor resultado exigiu adaptar a rede inteira ao novo domínio.

## 24. Significado de média não ponderada

Uma média não ponderada atribui a mesma importância a cada valor, independentemente do tamanho do grupo que ele representa. No macro-F1, calcula-se primeiro o F1 separadamente para cada uma das dez classes do EuroSAT e depois se faz a média aritmética desses dez resultados. Cada classe contribui com 1/10 da métrica, tanto uma classe com 2.000 imagens quanto uma classe com 3.000 imagens. Em uma média ponderada, ao contrário, classes com mais imagens teriam influência maior. Essa escolha permite avaliar se o modelo funciona bem em todas as categorias e impede que um desempenho elevado nas classes maiores esconda resultados ruins nas menores.

## 25. Métricas de custo computacional

**Latência por imagem** é o tempo médio necessário para o modelo processar uma imagem e produzir sua previsão. No benchmark principal foi destacada a latência exclusiva do modelo com batch 1, após aquecimento da GPU e sincronização CUDA. Valores menores são melhores. Foram medidos 2,402 ms para ResNet18, 4,478 ms para ResNet34 e 6,152 ms para ResNet50.

**Throughput** é a quantidade de imagens processadas por segundo. Valores maiores são melhores. Ele depende do tamanho do lote e não deve ser comparado entre medições com batches diferentes. Com batch 64, o forward processou aproximadamente 1.938,76 imagens/s na ResNet18, 1.089,18 na ResNet34 e 624,58 na ResNet50.

**Pico de memória do forward** é a maior quantidade de memória da GPU alocada durante a passagem das imagens pelo modelo em modo de inferência. Forward é o caminho que vai da entrada até a previsão, sem cálculo de gradientes e sem atualização dos pesos. Com batch 64, os picos exclusivos do forward foram 479,58 MB, 518,18 MB e 773,24 MB. Esse valor não é o consumo total do computador, não equivale à memória de treinamento e não inclui necessariamente todo o pipeline de carregamento e pré-processamento.

**Tamanho do checkpoint** é o espaço ocupado no disco pelo arquivo que armazena os pesos aprendidos e metadados necessários para reconstruir o modelo. Foram medidos 42,73 MB para ResNet18, 81,35 MB para ResNet34 e 90,06 MB para ResNet50. O tamanho do checkpoint afeta armazenamento, distribuição e carregamento, mas não é igual ao pico de memória da GPU, pois os pesos e as ativações precisam ser carregados e representados na memória durante a execução.

Em síntese, latência responde quanto tempo uma previsão leva; throughput, quantas imagens são processadas por segundo; pico de memória, quanta memória foi necessária no momento mais exigente da inferência; e tamanho do checkpoint, quanto espaço o modelo ocupa quando salvo. Essas métricas complementam acurácia e macro-F1 na análise de eficiência.

## 26. Divisão fixa dos dados e teste fechado

“Divisão estratificada por classe” significa que as imagens de cada uma das dez classes foram separadas individualmente nas proporções de 70% para treino, 15% para validação e 15% para teste. Isso preservou aproximadamente a mesma composição de classes nas três partições. A seed 42 funcionou como ponto inicial fixo para o embaralhamento pseudoaleatório: usando os mesmos dados, código e seed, os mesmos índices são produzidos.

Índice é a posição de uma imagem na lista interna do dataset. Em vez de copiar as 27.000 imagens para três novas pastas, o projeto salvou em `data\splits\eurosat_stratified_seed42.json` três listas de posições, indicando quais exemplos pertencem a treino, validação e teste. Persistir significa gravar essas listas no disco para reutilizá-las. O carregador valida que todos os 27.000 índices aparecem exatamente uma vez e estão dentro do intervalo permitido.

Todos os experimentos carregaram o mesmo JSON. Por isso, ResNet18, ResNet34 e ResNet50, inclusive nas seeds de treinamento 42, 123 e 2026, foram comparadas com as mesmas 18.900 imagens de treino e as mesmas 4.050 imagens de validação. As seeds de treinamento alteraram fatores como inicialização da nova camada, ordem dos lotes e aumentos aleatórios, mas não mudaram quais imagens pertenciam a cada partição. Assim, diferenças de métricas podem ser atribuídas principalmente à arquitetura e à aleatoriedade do treinamento, e não a uma divisão mais fácil ou mais difícil.

Manter o teste “sem consulta” significa não calcular suas métricas, não usá-lo para escolher arquitetura, hiperparâmetros, época ou checkpoint e não utilizá-lo no benchmark. Durante o desenvolvimento, a validação serviu para essas decisões. Somente após registrar em `MODEL_SELECTION.md` a regra e a escolha da ResNet18, o teste foi aberto para uma avaliação final única. Esse procedimento reduz o risco de adaptar decisões ao teste e produzir uma estimativa excessivamente otimista de generalização.

Em termos simples, as imagens de teste já estavam baixadas e seus índices já constavam no JSON, mas `train.py` não as carregou para treinar nem calculou previsões sobre elas. “Sem consulta” não significa que os arquivos foram apagados, ocultados do computador ou que não sabíamos suas classes; significa que deliberadamente não usamos o desempenho nessas imagens para decidir nada. A validação funcionou como um simulado consultado durante o desenvolvimento, enquanto o teste funcionou como uma prova final lacrada. Depois que arquitetura, regra e checkpoint estavam fixados, `evaluate.py` abriu essa partição uma única vez e produziu a estimativa final de generalização.

## 27. Roteiro para apresentação ao orientador

A apresentação deve começar pelo objetivo: comparar ResNet18, ResNet34 e ResNet50 na classificação de uso e cobertura do solo no EuroSAT RGB, considerando tanto macro-F1 quanto custo computacional. A pergunta é qual arquitetura mantém desempenho elevado com menor latência, memória e tamanho de checkpoint para uma futura aplicação.

Passos executados: definição do problema e das métricas; download do EuroSAT RGB pelo `torchvision`; criação da divisão estratificada fixa de 70/15/15 com split-seed 42; implementação de transformações distintas para treino e avaliação; construção do pipeline PyTorch com CUDA; execução da baseline ingênua; treinamento da ResNet18 pré-treinada como baseline experimental; ablações com backbone congelado, learning rate alternativo e treinamento do zero; treinamento de ResNet18, ResNet34 e ResNet50 com seeds 42, 123 e 2026; geração de métricas e gráficos na validação; benchmark controlado de inferência; aplicação e registro de uma regra de seleção antes do teste; avaliação final única da ResNet18; análise por classe e consolidação dos artefatos.

Configuração principal: 15 épocas, batch 64, learning rate 0,0001, weight decay 0,0001, cross-entropy, AdamW, `ReduceLROnPlateau`, early stopping com paciência cinco, pesos ImageNet e ajuste completo. O melhor checkpoint de cada execução foi salvo pela menor perda de validação.

Resultados das ablações em macro-F1: baseline ingênua 2,00%; ResNet18 congelada com LR 0,0001, 81,47%; congelada com LR 0,001, 90,08%; treinamento do zero, 96,13%; pré-treinada com ajuste completo na seed 42, 98,46%. Isso mostrou o benefício da transferência de aprendizado e da adaptação de toda a rede.

Médias de validação das três seeds: ResNet18 98,5113% de macro-F1; ResNet34 98,5454%; ResNet50 98,8644%. No benchmark com batch 1, as latências foram 2,402, 4,478 e 6,152 ms/imagem. Com batch 64, os throughputs foram 1.938,76, 1.089,18 e 624,58 imagens/s. Os checkpoints mediram 42,73, 81,35 e 90,06 MB.

A regra prévia considerou elegíveis os modelos até 0,5 ponto percentual abaixo do melhor macro-F1 e escolheu, entre eles, a menor latência com batch 1. As três arquiteturas foram elegíveis, e a ResNet18 foi selecionada. A ResNet50 teve somente 0,353 ponto percentual adicional de macro-F1 médio, mas latência 2,56 vezes maior e checkpoint 2,11 vezes maior.

No teste final único de 4.050 imagens, a ResNet18 obteve 98,6667% de acurácia e 98,6122% de macro-F1. Cultivo permanente foi a classe mais difícil, com F1 de 97,30%, principalmente por confusões com cultivo anual e vegetação herbácea. Mar/lago e floresta atingiram 99,78% e 99,67% de F1.

Conclusão a apresentar: a ResNet50 venceu em desempenho absoluto, mas a ResNet18 apresentou o melhor compromisso entre desempenho e eficiência. A contribuição do trabalho é o protocolo experimental reproduzível e a análise conjunta de qualidade e custo, não a criação de uma nova rede. Limitações: somente EuroSAT RGB, três seeds, uma configuração de hardware, apenas ResNets, divisão por imagem e ausência de validação externa, multiespectral e temporal. O próximo trabalho acadêmico é revisar a redação, referências, figuras e apresentação, sem alterar retrospectivamente a seleção já concluída.

## 28. Significado de congelamento da rede

Congelar uma rede significa impedir que determinados pesos sejam atualizados durante o treinamento. Na ResNet18 congelada deste projeto, todos os parâmetros do backbone pré-treinado no ImageNet receberam `requires_grad=False`, e somente a nova camada final `fc`, com dez saídas para o EuroSAT, permaneceu treinável. O código também manteve as estatísticas das camadas BatchNorm congeladas. Assim, o backbone funcionou como um extrator fixo de características e a camada final aprendeu a associar essas características às dez classes.

O congelamento reduz a quantidade de parâmetros treináveis, o cálculo de gradientes e o custo de ajuste, podendo ser útil com poucos dados ou recursos limitados. A desvantagem é que as representações aprendidas no ImageNet não se adaptam ao domínio das imagens de satélite. No projeto, a ResNet18 congelada obteve 81,47% de macro-F1 com learning rate 0,0001 e 90,08% com 0,001, enquanto o ajuste completo atingiu 98,46%. Portanto, congelar a rede foi uma ablação útil, mas não foi a melhor estratégia para o EuroSAT.
