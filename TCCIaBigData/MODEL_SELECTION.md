# Registro prévio da seleção do modelo

Data UTC: 2026-08-23T17:24:20.036202+00:00

Status do conjunto de teste: **não avaliado**.

## Regra de decisão

1. Considerar elegíveis as arquiteturas cuja média de macro-F1 fique até 0.50 ponto percentual abaixo da melhor média de validação.
2. Entre as elegíveis, escolher a arquitetura com menor latência média de inferência do modelo usando batch 1.
3. Usar o checkpoint da seed 42, definido previamente como representante uniforme das arquiteturas no benchmark.

## Resultado da aplicação da regra

Modelo selecionado: **ResNet18 pré-treinada com ajuste completo**.

Checkpoint selecionado: `D:\TCCIaBigData\outputs\baseline_resnet18\best_model.pt`.

Macro-F1 médio: 98.5113% ± 0.0936%.

Latência com batch 1: 2.402 ± 0.045 ms/imagem.

A melhor média absoluta pertence à ResNet50 (98.8644%), com vantagem de 0.353 ponto percentual sobre o modelo selecionado.

A ResNet50 apresentou latência 2.56 vezes maior e checkpoint 2.11 vezes maior que o modelo selecionado.

## Justificativa

A ResNet18 permaneceu dentro da margem prática predefinida em relação ao melhor macro-F1 e apresentou o menor custo de inferência. Portanto, ela foi selecionada para a avaliação final única no teste. Este registro foi produzido sem consultar a partição de teste e não deve ser alterado em função do resultado final.
