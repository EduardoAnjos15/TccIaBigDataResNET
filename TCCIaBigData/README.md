# Classificação do EuroSAT com ResNet e PyTorch

Projeto inicial de TCC para classificar uso e cobertura do solo em imagens de
satélite. A implementação usa uma ResNet pré-treinada no ImageNet e *transfer
learning* no conjunto EuroSAT.

## O que já está implementado

- download automático das 27.000 imagens do EuroSAT pelo `torchvision`;
- divisão estratificada, reproduzível e persistida em treino (70%), validação
  (15%) e teste (15%);
- aumento de dados apenas no treino;
- ResNet-18, ResNet-34 ou ResNet-50;
- treinamento completo ou apenas da camada classificadora;
- early stopping e redução automática da taxa de aprendizado;
- acurácia, precisão, recall, F1 por classe e matriz de confusão;
- salvamento do melhor modelo e do histórico experimental;
- baseline ingênua por classe majoritária;
- avaliação do teste isolada do treinamento;
- registro de parâmetros, tempo e pico de memória da GPU.

## Preparação do ambiente

É recomendado usar Python 3.12. No PowerShell:

```powershell
D:\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Na RTX 3060 configurada para este projeto, instale a distribuição CUDA separadamente:

```powershell
python -m pip install torch==2.12.1 torchvision==0.27.1 `
  --index-url https://download.pytorch.org/whl/cu126
```

O PyTorch não está no `requirements.txt` de propósito: isso evita substituir a
distribuição CUDA por uma versão somente CPU. Para outra máquina, consulte o
seletor oficial: <https://pytorch.org/get-started/locally/>.

## Divisão congelada dos dados

Na primeira execução com uma seed, o projeto cria, por exemplo:

```text
data/splits/eurosat_stratified_seed42.json
```

Esse arquivo contém os índices estratificados de treino, validação e teste. Todos
os experimentos reutilizam exatamente a mesma divisão com `--split-seed 42`.
O argumento `--seed` controla apenas a inicialização e a aleatoriedade do treino,
permitindo repetições estatísticas sem alterar os exemplos de validação e teste.

## Baseline ingênua

A baseline ingênua encontra a classe mais frequente no treino e prevê essa classe
para todas as imagens. Durante o desenvolvimento, execute somente na validação:

```powershell
python naive_baseline.py `
  --seed 42 `
  --evaluation-split val `
  --output-dir outputs/baseline_naive
```

Ela gera `val_metrics.json` e `val_confusion_matrix.png`.

## Baseline experimental

```powershell
python train.py `
  --architecture resnet18 `
  --epochs 15 `
  --batch-size 64 `
  --learning-rate 0.0001 `
  --seed 42 `
  --output-dir outputs/baseline_resnet18
```

Na primeira execução, o EuroSAT e os pesos pré-treinados serão baixados. Os
arquivos serão gravados em `data/` e os resultados no diretório informado.
O treinamento calcula métricas completas somente na validação e não acessa o teste.

Se não houver GPU ou memória suficiente, reduza o lote:

```powershell
python train.py --epochs 10 --batch-size 16
```

Para treinar somente a última camada da rede:

```powershell
python train.py --freeze-backbone --epochs 10
```

Para criar experimentos comparáveis, altere apenas a arquitetura e o diretório:

```powershell
python train.py --architecture resnet18 --output-dir outputs/resnet18
python train.py --architecture resnet34 --output-dir outputs/resnet34
python train.py --architecture resnet50 --output-dir outputs/resnet50
```

Cada execução gera:

- `best_model.pt`: pesos da melhor época segundo a perda de validação;
- `metrics.json`: configuração, histórico e métricas de validação;
- `learning_curves.png`: curvas de perda e acurácia de treino e validação.
- `validation_confusion_matrix.png`: matriz de confusão da validação.

Diretórios com resultados existentes não são sobrescritos. Escolha outro
`--output-dir` ou use `--overwrite` conscientemente.

## Avaliação final no teste

Use o teste somente depois de selecionar e congelar as configurações finais:

```powershell
python evaluate.py `
  --checkpoint outputs/baseline_resnet18/best_model.pt
```

Esse comando gera `test_metrics.json` e `test_confusion_matrix.png`. Para avaliar
a baseline ingênua no teste final:

```powershell
python naive_baseline.py `
  --seed 42 `
  --evaluation-split test `
  --output-dir outputs/baseline_naive
```

## Gráficos para o TCC

Gere ou atualize os gráficos sempre que um novo experimento terminar:

```powershell
python generate_plots.py --seed 42
```

O script utiliza somente resultados associados ao split estratificado atual e
ignora automaticamente execuções antigas ou incompatíveis. São produzidos PNGs
em 300 dpi e PDFs vetoriais:

- distribuição das classes em treino, validação e teste;
- matriz de confusão normalizada de cada configuração;
- F1-score por classe de cada configuração;
- comparação de acurácia e macro-F1 entre baselines e modelos;
- relação entre macro-F1 e tempo quando houver duas ou mais configurações.

Os gráficos globais são salvos em `outputs/plots/`; os gráficos específicos de
cada modelo ficam no próprio diretório do experimento. Barras de erro aparecem
automaticamente quando existem execuções repetidas da mesma configuração.

### Amostras visuais do EuroSAT

Para selecionar três imagens reais por classe na partição de treino e gerar um
painel reprodutível em PNG e PDF:

```powershell
python generate_dataset_samples.py `
  --partition train `
  --samples-per-class 3 `
  --split-seed 42 `
  --selection-seed 42
```

Os arquivos são salvos em `outputs/dataset_samples`. O manifesto JSON registra
os índices e caminhos das 30 imagens selecionadas. Além do painel geral, a pasta
`class_panels` contém uma figura PNG 300 dpi e uma PDF para cada classe, com a
legenda no topo e os três exemplos abaixo. A partição de teste não é aceita por
esse script, preservando o protocolo da avaliação final.

## Benchmark de inferência

Depois de concluir as repetições de treinamento, meça o custo de inferência dos
checkpoints da seed 42 usando somente a validação:

```powershell
python benchmark_inference.py `
  --batch-sizes 1 64 `
  --repeats 5 `
  --warmup-batches 20 `
  --model-iterations 100 `
  --no-download
```

O benchmark salva `benchmark_results.json` e `benchmark_summary.csv` em
`outputs/inference_benchmark`. Ele separa o tempo exclusivo do forward do modelo
do tempo do pipeline completo, mede o pico de memória de inferência e não acessa
a partição de teste.

## Gráficos de Pareto e seleção prévia

Com o benchmark oficial concluído, gere os gráficos de custo-benefício e registre
a decisão antes de acessar o teste:

```powershell
python generate_pareto_plots.py
```

O script combina a média e o desvio-padrão das três seeds com latência,
throughput, memória e tamanho do checkpoint. A regra padrão considera elegíveis
modelos até 0,5 ponto percentual abaixo do melhor macro-F1 e escolhe, entre eles,
o de menor latência com batch 1. A decisão é salva em `MODEL_SELECTION.md` e
`outputs/model_selection/selection_decision.json`. O script se recusa a criar o
registro caso encontre um `test_metrics.json`.

## Relatório após o teste final

Depois de executar uma única vez `evaluate.py` com o checkpoint previamente
selecionado, gere os gráficos finais sem executar nova inferência:

```powershell
python generate_final_test_report.py
```

O script valida o checkpoint contra `selection_decision.json` e usa somente os
JSONs já salvos para produzir matriz normalizada, F1 por classe, comparação entre
validação e teste, CSV por classe e relatório Markdown.

## Proposta de questão de pesquisa

> Qual é o impacto da profundidade da arquitetura residual e do aprendizado por
> transferência na classificação de uso e cobertura do solo no EuroSAT,
> considerando desempenho preditivo e custo computacional?

Para uma comparação justa, mantenha a mesma divisão (`--seed 42`), quantidade de
épocas, aumentos de dados e métrica principal. Recomenda-se relatar macro-F1 além
da acurácia.

## Próximas etapas do TCC

1. Executar um teste curto e validar todo o pipeline.
2. Treinar ResNet-18, 34 e 50 com três sementes diferentes.
3. Comparar pesos pré-treinados contra treinamento do zero (`--no-pretrained`).
4. Gerar gráficos, matriz de confusão e visualizações Grad-CAM.
5. Documentar limitações, erros mais comuns e custo computacional.
