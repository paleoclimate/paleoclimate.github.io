# Comparação Floegel × GeoTIFF

Este diretório contém o fluxo de **similaridade espacial** entre os mapas de referência publicados (Floegel) e os mapas gerados pelo projeto (GeoTIFF KNN+IDW). O script responsável é `compare_floegel.py`, na raiz do repositório.

Idades suportadas: **105 Ma** e **115 Ma**.

---

## Pré-requisitos

Antes de comparar, confira se você tem:

| Arquivo | Descrição |
|---------|-----------|
| `GENERATED_GEOTIFFS/105_ma_knn_idw_power4.0_gradient_sharp18.0_kdtree.tif` | GeoTIFF gerado para 105 Ma |
| `GENERATED_GEOTIFFS/115_ma_knn_idw_power4.0_gradient_sharp18.0_kdtree.tif` | GeoTIFF gerado para 115 Ma |
| `COMPARISON/inputs/105_floegel.png` | Recorte PNG do mapa Floegel (105 Ma) |
| `COMPARISON/inputs/115_floegel.png` | Recorte PNG do mapa Floegel (115 Ma) |
| `COMPARISON/inputs/legenda.png` | Recorte da legenda de cores do Floegel |

Se os GeoTIFFs ainda não existem, gere os mapas primeiro:

```bash
python render_map.py --power 4.0 --gradient-sharp 18.0 --kdtree --map 105 115
```

---

## Passo a passo

### 1. Gerar o render de referência e o GCP Picker

Na raiz do repositório:

```bash
python compare_floegel.py render-reference
```

Isso cria ou atualiza:

- `COMPARISON/reference_render_105.png` e `reference_render_115.png` — raster do GeoTIFF com as mesmas cores do mapa publicado + linha de costa
- `COMPARISON/gcp_picker.html` — ferramenta interativa para marcar pontos de controle (GCPs)

> **Importante:** rode este comando de novo sempre que regenerar os mapas com `render_map.py`. O picker usa o GeoTIFF mais recente; se o `.tif` mudou, o render de referência precisa ser refeito.

Para processar só uma idade:

```bash
python compare_floegel.py render-reference --age 105
```

---

### 2. Marcar os pontos de controle (GCPs)

1. Abra `COMPARISON/gcp_picker.html` no navegador (duplo clique ou servidor local, ex.: Live Server).
2. Selecione a idade (105 Ma ou 115 Ma) no menu superior.
3. Para cada par de pontos:
   - clique **uma vez** no mapa Floegel (painel esquerdo);
   - clique no ponto equivalente no render do GeoTIFF (painel direito).
4. Repita **6 a 12 vezes** por idade (mínimo 4 pares). Prefira cantos, extremidades de continentes e marcos fáceis de identificar nos dois lados.
5. Clique em **Exportar gcp_&lt;idade&gt;.json** e salve o arquivo.

Coloque os JSON exportados em `COMPARISON/`:

```
COMPARISON/
  gcp_105.json
  gcp_115.json
```

Se já existirem arquivos antigos, **substitua** pelos novos. Os pontos são usados para deformar (warp) a imagem Floegel até a grade do GeoTIFF antes da comparação.

> Se você regenerou o render de referência após uma atualização do código, **remarque** os GCPs. Pontos marcados em versões antigas do render podem estar espelhados e invalidam o warp.

---

### 3. Rodar a comparação

Com os JSON na pasta `COMPARISON/`:

```bash
python compare_floegel.py compare
```

Isso gera, para cada idade:

| Arquivo | Conteúdo |
|---------|----------|
| `metrics_<idade>.csv` | IoU por classe, acurácia geral, Cohen's kappa |
| `classes_<idade>.png` | Painel: GeoTIFF × Floegel deformado × matriz de confusão |
| `warped_floegel_<idade>.png` | Floegel após warp |
| `comparison_report_<idade>.html` | Relatório visual completo |
| `index_comparison.html` | Índice com links para os relatórios |

Abra `COMPARISON/index_comparison.html` no navegador para ver os resultados.

Para comparar só uma idade:

```bash
python compare_floegel.py compare --age 115
```

---

## Comandos auxiliares

| Comando | Quando usar |
|---------|-------------|
| `python compare_floegel.py pick-gcps` | Regenerar só o `gcp_picker.html` (por exemplo, após trocar os PNGs do Floegel em `inputs/`) |
| `python compare_floegel.py pick-gcps --no-refresh` | Regenerar o HTML sem re-renderizar a partir do GeoTIFF (só se você tem certeza de que o render já está atualizado) |
| `python compare_floegel.py --suffix <sufixo> ...` | Usar GeoTIFFs com outro sufixo de parâmetros (default: `knn_idw_power4.0_gradient_sharp18.0_kdtree`) |

---

## Estrutura de pastas

```
COMPARISON/
  README.md                 ← este arquivo
  inputs/
    105_floegel.png         ← mapas Floegel (entrada manual)
    115_floegel.png
    legenda.png
  gcp_picker.html           ← gerado (passo 1)
  reference_render_105.png  ← gerado (passo 1)
  reference_render_115.png
  gcp_105.json              ← exportado manualmente (passo 2)
  gcp_115.json
  metrics_*.csv             ← gerado (passo 3)
  comparison_report_*.html
  index_comparison.html
  classes_*.png
  warped_floegel_*.png
```

---

## Resumo rápido

```bash
# 0. (se necessário) gerar mapas
python render_map.py --power 4.0 --gradient-sharp 18.0 --kdtree --map 105 115

# 1. render + picker
python compare_floegel.py render-reference

# 2. abrir COMPARISON/gcp_picker.html, marcar pontos, salvar gcp_105.json e gcp_115.json em COMPARISON/

# 3. comparar
python compare_floegel.py compare
```
