# Fontes públicas do corpus unificado

O corpus de 12,7 milhões de consultas do artigo é compilado a partir das dez
fontes públicas abaixo (Tabela 1 do artigo). Baixe cada fonte para
`code/datasets/raw/` e execute `python code/compilation/compile.py` a partir da
raiz do repositório. A deduplicação cross-source usa ordem de prioridade fixa e
seed 42; cinco fontes são totalmente subsumidas por fontes de maior prioridade
e contribuem com zero linhas, como reportado no artigo.

| # | Fonte | Tipo | Linhas brutas | Onde obter | Licença |
|---|-------|------|--------------:|------------|---------|
| 1 | RbSQLi (Mullick et al., 2026) | Sintética (rule-based) | 10.190.450 | Mendeley Data v5, DOI [10.17632/xz4d5zj5yw.5](https://data.mendeley.com/datasets/xz4d5zj5yw/5) | CC BY 4.0 |
| 2 | Superviz25-SQL (Quetel et al., 2025) | Semi-sintética | 3.688.977 | Zenodo v2, DOI [10.5281/zenodo.17086037](https://zenodo.org/records/17086037) | MIT |
| 3 | SQL Injection Dataset (Kaldi, 2024) | Mista (inclui texto em linguagem natural) | 163.476 | [Kaggle: ayahkhaldi/sql-injection-dataset](https://www.kaggle.com/datasets/ayahkhaldi/sql-injection-dataset) | ver página |
| 4 | BCCC-SFU-SQLInj-2023 (Issakhani et al., 2023) | Adversarial | 11.011 | [BCCC, York University / SFU](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/sql-injection-attack-bccc-sfu-sqlinj-2023/) | ver página |
| 5 | SQL Injection Dataset (Rayten, 2024) | Sintética | 122.982 | [Kaggle: rayten/sql-injection-dataset](https://www.kaggle.com/datasets/rayten/sql-injection-dataset) | Apache 2.0 |
| 6 | Modified SQL (Sajid, 2021)* | Sintética | 30.919 | [Kaggle: sajid576/sql-injection-dataset](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset) | ver página |
| 7 | SQL-data (Mathew, 2021)* | Mista | 37.961 | [GitHub: ajinmathew/SQL-data](https://github.com/ajinmathew/SQL-data) | ver página |
| 8 | Biggest SQLi (Yu, 2023)* | Sintética | 148.326 | [Kaggle: gambleryu/biggest-sql-injection-dataset](https://www.kaggle.com/datasets/gambleryu/biggest-sql-injection-dataset) | ver página |
| 9 | SQLi Extended (Trinity, 2022)* | Sintética | 109.518 | [Kaggle: alextrinity/sqlinjectionextend](https://www.kaggle.com/datasets/alextrinity/sqlinjectionextend) | CC0 |
| 10 | SQLiV3 (Hussain, 2021)* | Sintética | 30.609 | [Kaggle: syedsaqlainhussain/sql-injection-dataset](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset) | ver página |

\* Totalmente subsumida após a deduplicação cross-source; contribui com zero
linhas para o corpus final, mas é necessária para reproduzir exatamente o
processo de compilação descrito no artigo. Para uma reconstrução aproximada
apenas das linhas contribuintes, as fontes 1 a 5 são suficientes.

Total esperado após compilação: aproximadamente **12.732.002 consultas
rotuladas** (75,5% benignas, 24,5% maliciosas), com split estratificado
70/15/15 (treino/validação/teste) por par (fonte, rótulo), seed 42. Valide com
`python code/datasets/sanity_check.py`.

As fontes não são redistribuídas neste repositório; cada uma mantém a licença
original indicada em sua página. As fontes hospedadas no Kaggle exigem conta
gratuita para download.
