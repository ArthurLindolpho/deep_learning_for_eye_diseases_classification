# Deep Learning para Classificação de Doenças Oculares

Trabalho final da disciplina de **Visão Computacional** (UERJ). Comparamos três arquiteturas de deep learning na classificação de doenças oculares a partir de imagens de fundo de olho: uma **CNN clássica**, a **VGG19** e a **EfficientNetB0**.

Além do resultado, o objetivo foi entender a arquitetura de cada modelo, suas vantagens e desvantagens, e como avaliar um classificador de imagens com rigor.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

---

## Resultados em resumo

Todos os experimentos foram repetidos com **5 sementes diferentes**. Os valores são média ± desvio padrão.

| Modelo | Dataset | Classes | Acurácia | F1 (ponderado) |
|---|---|---|---|---|
| VGG19 | Eye Diseases Classification | 4 | **73,8% ± 2,6** | 73,3% ± 2,7 |
| VGG19 | Retinal Fundus Images | 11 | 66,5% ± 0,9 | 61,4% ± 1,1 |
| CNN clássica | Retinal Fundus Images | 11 | 63,4%¹ | 61,2%¹ |
| EfficientNetB0 | Eye Diseases Classification | 4 | script de treino disponível² | — |

¹ Calculado a partir da matriz de confusão média das 5 rodadas, salva no notebook.
² O script [`EfficientNet (1).py`](EfficientNet%20(1).py) treina o modelo com 5 sementes e salva as métricas em CSV. Os resultados não estão versionados no repositório.

### O principal aprendizado: a média esconde as classes difíceis
No dataset de 11 classes, a CNN tem 63% de acurácia geral, mas o desempenho muda muito de uma doença para outra:

| Classe | Imagens de treino | F1 |
|---|---|---|
| Fundo normal | 2.641 | **0,90** |
| Catarata | 1.369 | 0,80 |
| Miopia patológica | 1.142 | 0,75 |
| Retinopatia diabética moderada | 4.982 | 0,72 |
| ... | | |
| DMRI seca | 1.276 | 0,37 |
| Retinopatia diabética leve | 2.294 | 0,32 |
| Retinopatia diabética proliferativa | 1.295 | 0,30 |
| DMRI úmida | 545 | **0,01** |

- **Classe com menos dados:** a DMRI úmida (degeneração macular relacionada à idade) tem apenas 545 imagens de treino, e o modelo praticamente não a reconhece.
- **Graus de uma mesma doença:** os estágios da retinopatia diabética (leve, moderada, severa e proliferativa) se confundem entre si, porque são graus vizinhos.

O F1 macro, que dá o mesmo peso a todas as classes, é **0,53**, bem abaixo dos 0,61 do F1 ponderado. Por isso avaliar cada classe separadamente é essencial.

---

## Dados

| Dataset | Classes | Divisão |
|---|---|---|
| [Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data) | 4: catarata, retinopatia diabética, glaucoma e normal | 80% treino / 20% teste, estratificada a cada semente |
| [Retinal Fundus Images](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data) | 11: DMRI seca e úmida, 4 graus de retinopatia diabética, glaucoma, catarata, retinopatia hipertensiva, miopia patológica e normal | treino (~20 mil) / validação (433) / teste (1.236), já separados |

Os datasets não estão no repositório. Faça o download pelo Kaggle.

## Metodologia

1. **Pré-processamento:**
   - imagens redimensionadas para 224 × 224 e normalizadas;
   - aumento de dados com rotações e filtros, para ampliar a base e melhorar a generalização.
2. **Modelos:**
   - **CNN clássica:** uma camada convolucional (32 filtros), max pooling, global average pooling e uma camada densa. É um baseline leve, com apenas ~2,3 mil parâmetros.
   - **VGG19:** a base convolucional da VGG19 com um classificador denso no topo (512 → 128 → 32 neurônios, com dropout de 0,5).
   - **EfficientNetB0:** treinada do zero, com global average pooling e softmax.
3. **Treino:**
   - otimizador Adam, até 50 épocas;
   - early stopping na CNN, monitorando a perda de validação.
4. **Avaliação:**
   - 5 execuções com sementes diferentes, reportando média e desvio padrão;
   - acurácia, precisão, recall, F1 e matriz de confusão.

## Limitações e próximos passos
- **Transfer learning:** nos experimentos, a base da VGG19 foi carregada sem pesos pré-treinados (`weights=None`) e congelada (`trainable=False`), então só o classificador do topo aprendeu. Usar os pesos do ImageNet, e liberar o ajuste das últimas camadas (fine-tuning), deve melhorar bastante o resultado.
- **Aumento de dados e divisão:** no dataset de 4 classes, a divisão foi feita depois do aumento de dados. Assim, versões rotacionadas da mesma imagem podem cair no treino e no teste. O ideal é separar antes e aumentar só o treino.
- **Classes desbalanceadas:** pesos por classe ou mais exemplos das classes raras, principalmente a DMRI úmida.
- **Avaliação:** reportar o F1 macro e as métricas por classe, não só as médias ponderadas.

## Estrutura do repositório

| Arquivo | Conteúdo |
|---|---|
| `cnn.ipynb` | CNN clássica nos dois datasets e matriz de confusão média |
| `VGG_19.ipynb` | VGG19 nos dois datasets, 5 rodadas cada |
| `EfficientNet (1).py` | Script de treino da EfficientNetB0, com logs por época |
| `requirements.txt` | Dependências |

## Como executar

```bash
pip install -r requirements.txt
```

Baixe os datasets do Kaggle, ajuste os caminhos das pastas no início de cada notebook ou script e execute. Ter uma GPU é recomendado.

## Equipe

- Arthur Tavares Lindolpho
- Estevão dos Santos Mello
- Cauã Gomes
- Guilherme Alves
- Marcelo Ângelo
- Nicolas Ouverney
