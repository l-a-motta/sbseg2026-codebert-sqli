# Fontes públicas do corpus unificado

O corpus de 12,7 milhões de consultas do artigo é compilado a partir das dez
fontes públicas abaixo (Tabela 1 do artigo). Esta página documenta **o que
baixar, com que nome salvar cada arquivo e como validar o resultado**.

## Como preparar `code/datasets/raw/`

1. Baixe cada fonte da coluna "Onde obter" (a **ordem de download é
   irrelevante** — `compile.py` processa as fontes em ordem fixa de prioridade
   de deduplicação, independente da ordem em que foram baixadas).
2. Salve/renomeie cada arquivo baixado **exatamente** com o nome da coluna
   "Salvar como" (o nome original do arquivo é o próprio nome após o prefixo
   `DSxx_`; a renomeação apenas adiciona o prefixo). Todos os arquivos ficam
   diretamente em `code/datasets/raw/`, sem subpastas.
3. Execute `python code/compilation/compile.py` a partir da raiz do
   repositório. Arquivos ausentes são **pulados com um aviso** no log — útil
   para reconstruções parciais —, mas a reprodução exata do corpus do artigo
   exige todos os arquivos listados.
4. Valide com `python code/datasets/sanity_check.py`.

| # | Fonte | Tipo | Linhas brutas | Salvar como | Onde obter | Licença |
|---|-------|------|--------------:|-------------|------------|---------|
| 1 | RbSQLi (Mullick et al., 2026) | Sintética (rule-based) | 10.190.450 | `DS01_rbsqli_dataset.csv` | Mendeley Data v5, DOI [10.17632/xz4d5zj5yw.5](https://data.mendeley.com/datasets/xz4d5zj5yw/5) | CC BY 4.0 |
| 2 | Superviz25-SQL (Quetel et al., 2025) | Semi-sintética | 3.688.977 | `DS02_dataset.csv` | Zenodo v2, DOI [10.5281/zenodo.17086037](https://zenodo.org/records/17086037) | MIT |
| 3 | SQL Injection Dataset (Kaldi, 2024) | Mista (inclui texto em linguagem natural) | 163.476 | `DS03_Train.csv`, `DS03_Validation.csv`, `DS03_Test.csv` | [Kaggle: ayahkhaldi/sql-injection-dataset](https://www.kaggle.com/datasets/ayahkhaldi/sql-injection-dataset) | ver página |
| 4 | BCCC-SFU-SQLInj-2023 (Issakhani et al., 2023) | Adversarial | 11.011 | `DS04_BCCC-SFU-SQLInj-2023.csv` | [BCCC, York University / SFU](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/sql-injection-attack-bccc-sfu-sqlinj-2023/) | ver página |
| 5 | SQL Injection Dataset (Rayten, 2024) | Sintética | 122.982 | `DS11_testingdata.csv`, `DS12_trainingdata.csv` | [Kaggle: rayten/sql-injection-dataset](https://www.kaggle.com/datasets/rayten/sql-injection-dataset) | Apache 2.0 |
| 6 | Modified SQL (Sajid, 2021)* | Sintética | 30.919 | `DS06_Modified_SQL_Dataset.csv` | [Kaggle: sajid576/sql-injection-dataset](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset) | ver página |
| 7 | SQL-data (Mathew, 2021)* | Mista | 37.961 | `DS07_sqliv2.csv`, `DS08_sqli.csv` | [GitHub: ajinmathew/SQL-data](https://github.com/ajinmathew/SQL-data) | ver página |
| 8 | Biggest SQLi (Yu, 2023)* | Sintética | 148.326 | `DS09_clean_sql_dataset.csv` | [Kaggle: gambleryu/biggest-sql-injection-dataset](https://www.kaggle.com/datasets/gambleryu/biggest-sql-injection-dataset) | ver página |
| 9 | SQLi Extended (Trinity, 2022)* | Sintética | 109.518 | `DS10_sqli-extended.csv` | [Kaggle: alextrinity/sqlinjectionextend](https://www.kaggle.com/datasets/alextrinity/sqlinjectionextend) | CC0 |
| 10 | SQLiV3 (Hussain, 2021)* | Sintética | 30.609 | `DS13_SQLiV3.csv` | [Kaggle: syedsaqlainhussain/sql-injection-dataset](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset) | ver página |

\* Totalmente subsumida após a deduplicação cross-source; contribui com zero
linhas para o corpus final, mas é necessária para reproduzir exatamente o
processo de compilação descrito no artigo. Para uma reconstrução aproximada
apenas das linhas contribuintes, as fontes 1 a 5 são suficientes.

A numeração `DSxx_` dos nomes de arquivo segue o catálogo interno de datasets
do projeto e não é contígua nem coincide com a coluna `#` da tabela (que segue
a Tabela 1 do artigo): fontes distribuídas pré-divididas em múltiplos arquivos
recebem um identificador por arquivo (fonte 3 → DS03 com três arquivos; fonte 5
→ DS11 e DS12; fonte 7 → DS07 e DS08).

**Arquivo opcional — `DS05_SQLI_Dataset.csv`:** o dataset de features
[SQL Injection Attack Dataset (Alyasiri & Al-Araji, Mendeley Data, DOI
10.17632/mmc4sdmnrc.1)](https://data.mendeley.com/datasets/mmc4sdmnrc/1)
(salvar `SQLI_Dataset.csv` como `DS05_SQLI_Dataset.csv`) gera o arquivo
auxiliar `compiled_features.csv`. Ele **não** integra o corpus BERT nem é
usado por nenhuma das claims do artigo (os baselines de ML extraem as 18
features diretamente do texto das consultas); se ausente, `compile.py` pula
essa etapa com um aviso.

A deduplicação cross-source usa ordem de prioridade fixa e seed 42; cinco
fontes são totalmente subsumidas por fontes de maior prioridade e contribuem
com zero linhas, como reportado no artigo.

Total esperado após compilação: aproximadamente **12.732.002 consultas
rotuladas** (75,5% benignas, 24,5% maliciosas), com split estratificado
70/15/15 (treino/validação/teste) por par (fonte, rótulo), seed 42. Valide com
`python code/datasets/sanity_check.py`.

As fontes não são redistribuídas neste repositório; cada uma mantém a licença
original indicada em sua página. As fontes hospedadas no Kaggle exigem conta
gratuita para download.
