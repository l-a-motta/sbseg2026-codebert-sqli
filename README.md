# CodeBERT Detection Pipeline for SQL Injection Detection: A Comparison with other ML Models

Artefato do artigo completo aceito na Trilha Principal (Sistemas) do XXVI Simpósio Brasileiro de Cibersegurança (SBSeg 2026).

**Resumo:** SQL injection (SQLi) continua sendo uma vulnerabilidade prevalente em aplicações web, e as abordagens de detecção existentes frequentemente falham em generalizar para payloads ofuscados e longos. O artigo apresenta o **CodeBERT-SQLi**, um pipeline de detecção baseado em CodeBERT que adapta modelos transformer a consultas de comprimento arbitrário por meio de tokenização com janela deslizante (sliding window) e agregação por votação majoritária. Compilamos e padronizamos um corpus de SQLi em larga escala a partir de dez fontes públicas, produzindo um dataset unificado de 12,7 milhões de consultas rotuladas, e conduzimos uma avaliação comparativa envolvendo o CodeBERT-SQLi, BERT e CodeBERT sem fine-tuning (linear probing) e baselines tradicionais de ML (Regressão Logística, Random Forest e SVM Linear). Os experimentos na partição de teste completa de 1,9 milhão de consultas mostram que o CodeBERT-SQLi atinge MCC = 0,947 e AUC = 0,971, superando todos os métodos de comparação.

Este repositório contém o código-fonte completo do pipeline: compilação do corpus, fine-tuning, avaliação e comparação. O corpus compilado e o modelo treinado não são redistribuídos aqui; o corpus é integralmente reconstruível a partir das fontes públicas documentadas em [`code/datasets/SOURCES.md`](code/datasets/SOURCES.md) usando os scripts deste repositório.

# Estrutura do repositório

```
.
├── README.md                     # este arquivo
├── LICENSE                       # licença MIT
├── requirements.txt              # dependências Python com versões fixadas
└── code/
    ├── compilation/
    │   ├── compile.py            # compila o corpus unificado (normalização de schema,
    │   │                         #   deduplicação cross-source, split 70/15/15, seed 42)
    │   └── explore_raw.py        # exploração e validação das fontes brutas
    ├── datasets/
    │   ├── SOURCES.md            # as dez fontes públicas: links, DOIs e licenças
    │   ├── fixture/
    │   │   └── fixture_dataset.csv  # amostra de 109 linhas com o schema completo (teste mínimo)
    │   ├── build_fixture.py      # gera o fixture a partir do corpus compilado
    │   ├── sanity_check.py       # checagens de integridade do corpus compilado
    │   └── enrich_metadata.py    # preenche metadados auxiliares por fonte
    ├── fine-tuning/
    │   ├── train.py              # fine-tuning do CodeBERT-SQLi (sliding window + weighted loss)
    │   └── train_colab.ipynb     # mesmo treino empacotado para Google Colab (GPU)
    ├── eval/
    │   ├── evaluate.py           # avalia modelos fine-tuned na partição de teste (majority vote)
    │   ├── linear_probe.py       # probes lineares BERT-base vs. CodeBERT-base (encoder congelado)
    │   ├── evaluate_baselines.py # Regressão Logística, Random Forest e SVM Linear (18 features)
    │   └── eval_colab.ipynb      # avaliação empacotada para Google Colab (GPU)
    └── comparison/
        └── compare.py            # consolida resultados e gera as tabelas/figuras do artigo
```

Todos os scripts assumem execução a partir da **raiz do repositório**. Os parâmetros de execução (caminhos de dataset e de saída, seed, batch size, limites de linhas, modelo) são **configuráveis por flags de linha de comando** — execute qualquer script com `--help` para a lista completa. Os valores padrão (declarados como constantes documentadas no topo de cada arquivo) reproduzem exatamente as execuções do artigo, de modo que rodar sem flags equivale à configuração publicada. Cada execução grava um log detalhado em `code/<estágio>/logs/`.

# Selos considerados

Os selos considerados pelos autores são:

- **Selo D — Artefatos Disponíveis:** código completo neste repositório público e estável, com documentação e licença.
- **Selo F — Artefatos Funcionais:** dependências documentadas com versões fixadas e teste mínimo executável em CPU em poucos minutos (seção "Teste mínimo").
- **Selo S — Artefatos Sustentáveis:** código modular por estágio, parametrização externa via flags de linha de comando (`--help` em cada script) com padrões que reproduzem o artigo, nomes autoexplicativos e logging detalhado em todas as etapas.

A reprodução integral da claim principal (fine-tuning em 8,86M de linhas) é documentada, mas exige aproximadamente 108 horas de GPU A100 40 GB; por isso o Selo R não é reivindicado. A seção "Experimentos" indica reproduções em escala reduzida para cada claim.

# Informações básicas

**Teste mínimo (fixture):**
- Qualquer máquina x86-64 com Python 3.11 ou superior, 8 GB de RAM, sem GPU.
- Acesso à internet apenas para instalar dependências (e baixar os encoders da Hugging Face, no caso do probe).

**Experimentos completos do artigo:**
- Disco: aproximadamente 50 GB para as fontes brutas e o corpus compilado (12,7M de linhas).
- Fine-tuning: GPU NVIDIA A100 40 GB, aproximadamente 108 horas (executado via `train_colab.ipynb` no Google Colab).
- Avaliação do modelo fine-tuned na partição de teste completa: aproximadamente 2,4 horas em A100.
- Baselines de ML: CPU, minutos.

# Dependências

Python 3.11+. Dependências com as versões testadas em `requirements.txt`:

```
pandas, numpy, scikit-learn, torch, transformers, matplotlib, psutil
```

Os scripts de probe e fine-tuning baixam automaticamente os encoders públicos `microsoft/codebert-base` e `bert-base-uncased` da Hugging Face na primeira execução. Não há dependência de serviços pagos ou credenciais.

# Preocupações de segurança

O corpus e o fixture contêm **payloads reais e sintéticos de ataques de SQL injection como texto inerte**, usados exclusivamente como dados de treinamento e avaliação de classificadores. Nenhum script deste repositório executa consultas contra bancos de dados ou aplicações; nada é enviado a serviços externos. Os avaliadores não devem copiar payloads do corpus para sistemas em produção. O propósito do artefato é estritamente defensivo (detecção).

# Instalação

```bash
git clone https://github.com/l-a-motta/sbseg2026-codebert-sqli.git
cd sbseg2026-codebert-sqli
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

# 🟢 Teste mínimo

> **Escopo:** CPU, poucos minutos, sem GPU e sem downloads externos — usa apenas o fixture de 109 linhas incluído no repositório. É tudo o que é necessário para verificar a funcionalidade do artefato (Selo F). Não confundir com os **Experimentos completos** da seção seguinte, que reproduzem as claims do artigo em escala real.

O fixture de 109 linhas reproduz o schema completo do corpus compilado e permite exercitar o pipeline de ponta a ponta em CPU. A partir da raiz do repositório:

```bash
# 1. Posiciona o fixture no caminho esperado pelos scripts
mkdir -p code/datasets/compiled
cp code/datasets/fixture/fixture_dataset.csv code/datasets/compiled/compiled_dataset.csv

# 2. Checagem de integridade do dataset
python code/datasets/sanity_check.py

# 3. Baselines de ML (Regressão Logística, Random Forest, SVM) — segundos, CPU
python code/eval/evaluate_baselines.py
```

Saída esperada do passo 3: métricas (MCC, AUC, F1, acurácia, precisão, revocação) impressas no console para os três baselines e CSVs de predições em `code/eval/results/`. Os valores absolutos com o fixture não são comparáveis aos do artigo (109 linhas vs. 12,7M); o teste demonstra funcionalidade.

Opcional (baixa os dois encoders da Hugging Face, alguns minutos em CPU):

```bash
python code/eval/linear_probe.py --probe-train-rows 50
```

O passo 1 (cópia do fixture) atende os scripts que usam o caminho padrão do corpus; alternativamente, qualquer script aceita `--dataset-path code/datasets/fixture/fixture_dataset.csv` diretamente, sem a cópia.

---

# 🔴 Experimentos completos

> **Escopo:** reprodução das claims do artigo em escala real — requer baixar ~50 GB de fontes brutas e, para a Claim 1, ~108 horas de GPU A100. **Nenhum passo desta seção é necessário para o teste mínimo acima.**

Pré-requisito comum: baixar as dez fontes públicas (links, nomes de arquivo esperados e instruções em `code/datasets/SOURCES.md`) para `code/datasets/raw/` e compilar o corpus:

```bash
python code/compilation/compile.py   # gera code/datasets/compiled/compiled_dataset.csv
python code/datasets/sanity_check.py # valida contagens: ~12,7M linhas, 75,5% benignas
```

**Claim 1 — CodeBERT-SQLi atinge MCC = 0,947 e AUC = 0,971 na partição de teste de 1,9M de consultas** (Tabela 2 do artigo). Reprodução: `python code/fine-tuning/train.py --preset codebert-sqli` (ou `train_colab.ipynb` em GPU; ~108 h em A100) seguido de `python code/eval/evaluate.py`. Reprodução em escala reduzida: `python code/fine-tuning/train.py --preset codebert-sqli --max-rows-per-split 100000` reduz o treino para horas em GPU modesta, com métricas inferiores às reportadas.

**Claim 2 — O pré-treinamento em código (ΔMCC = 0,059) contribui mais que o fine-tuning específico (ΔMCC = 0,026)** (Seção 4). Reprodução: `python code/eval/linear_probe.py` treina probes lineares idênticos sobre BERT-base e CodeBERT-base congelados (50 mil linhas do split de treino) e os avalia na partição de teste completa; a diferença entre os dois probes isola o efeito do pré-treinamento. Executável em GPU única em poucas horas.

**Claim 3 — Os baselines tradicionais de ML ficam entre MCC 0,743 e 0,754** (Tabela 2 do artigo). Reprodução: `python code/eval/evaluate_baselines.py` (amostra de treino de 100 mil linhas — ajustável via `--max-train-rows` —, avaliação na partição de teste completa; CPU, minutos com o corpus compilado).

Consolidação: `python code/comparison/compare.py` lê os CSVs de `code/eval/results/` e gera as tabelas e figuras comparativas do artigo em `code/comparison/output/`.

Todos os estágios usam a seed fixa 42 como padrão (ajustável via `--seed`), garantindo splits e amostragens idênticos aos do artigo quando executados sem flags.

# LICENSE

Este projeto é licenciado sob a licença MIT — ver o arquivo [LICENSE](LICENSE). As fontes de dados listadas em `code/datasets/SOURCES.md` mantêm suas licenças originais, indicadas naquele arquivo.
