# Perch Mapping Diagnostics

## 1. Summary

The W0.25 blend notebook wrote `perch_mapping_diagnostics.csv` with one row per
BirdCLEF submission column.

| Mapping status | Columns |
|---|---:|
| Exact Perch scientific-name match | **203** |
| Unmapped | **31** |
| Total submission columns | **234** |

Exact Perch mapping covers **86.8%** of the submission columns.

## 2. Unmapped Columns

Most unmapped columns are anonymous insect sonotypes rather than named taxa.

| Group | Count | Notes |
|---|---:|---|
| `Insect son01` through `Insect son25` | **25** | No real genus/species string, so genus proxy is unsafe |
| Named taxa | **6** | Possible proxy candidates if Perch metadata has genus matches |

Named unmapped taxa:

| Primary label | Scientific name | Genus |
|---|---|---|
| `116570` | `Caiman yacare` | `Caiman` |
| `1491113` | `Adenomera guarani` | `Adenomera` |
| `1595929` | `Lysapsus limellum` | `Lysapsus` |
| `25073` | `Chiasmocleis mehelyi` | `Chiasmocleis` |
| `516975` | `Sapajus cay` | `Sapajus` |
| `74580` | `Mico melanurus` | `Mico` |

## 3. Interpretation

The diagnostics explain why the exact-mapped blend already works well: Perch
contributes directly to most target columns. The remaining mapping gap is
concentrated in anonymous insect sonotypes, where scientific-name matching and
genus proxying do not apply.

This means a broad proxy-mapping experiment is unlikely to help much and could
hurt if it injects unrelated Perch logits into sonotype columns.

## 4. Recommendation

Do **not** proxy all unmapped columns.

The only reasonable proxy experiment is narrow:

1. Leave the 25 insect sonotype columns unchanged from SED.
2. For the 6 named unmapped taxa, search Perch metadata for same-genus labels.
3. Use a small proxy weight only for those 6 columns.
4. Keep the existing exact-mapped blend behavior unchanged.

If a narrow proxy variant is tested, keep it separate from the protected
`07_onnx_perch_sed_blend.ipynb` champion.

