# Soundscape Calibration Diagnostics

## 1. Summary

`soundscape_blend_calibration.csv` from
`10_onnx_perch_sed_soundscape_calibrated.ipynb` explains the move from
**0.892** to **0.893**.

| Metric | Value |
|---|---:|
| Submission columns | **234** |
| Exact-mapped classes | **203** |
| Proxy classes | **6** |
| SED-only sonotype classes | **25** |
| Classes with changed weights | **22** |
| Changed exact classes | **22** |
| Changed proxy classes | **0** |

The calibration gain came from exact-mapped classes, not from proxy classes.

## 2. Weight Distribution

| Best weight | Class count |
|---:|---:|
| 0.00 | 25 |
| 0.05 | 14 |
| 0.10 | 2 |
| 0.15 | 181 |
| 0.20 | 1 |
| 0.25 | 1 |
| 0.30 | 10 |

Most classes kept the default exact-mapped weight of **0.15**. Only **22**
classes moved away from their default.

## 3. Changed Classes

Changed classes with at least 10 positives:

| Label | Positives | Default | Best | AP gain |
|---|---:|---:|---:|---:|
| `bufpar` | 14 | 0.15 | 0.30 | 0.1139 |
| `orwpar` | 13 | 0.15 | 0.05 | 0.0765 |
| `chvcon1` | 35 | 0.15 | 0.05 | 0.0537 |
| `whtdov` | 63 | 0.15 | 0.30 | 0.0396 |
| `chacha1` | 65 | 0.15 | 0.05 | 0.0274 |
| `litnig1` | 37 | 0.15 | 0.05 | 0.0248 |
| `undtin1` | 43 | 0.15 | 0.25 | 0.0198 |
| `compau` | 38 | 0.15 | 0.30 | 0.0162 |
| `66971` | 149 | 0.15 | 0.30 | 0.0145 |
| `65380` | 333 | 0.15 | 0.30 | 0.0139 |
| `24279` | 173 | 0.15 | 0.30 | 0.0072 |
| `22967` | 155 | 0.15 | 0.10 | 0.0015 |
| `517063` | 313 | 0.15 | 0.30 | 0.0003 |
| `24321` | 172 | 0.15 | 0.10 | 0.0000 |

Changed classes with fewer than 10 positives:

| Label | Positives | Default | Best | AP gain |
|---|---:|---:|---:|---:|
| `fusfly1` | 3 | 0.15 | 0.30 | 0.5849 |
| `magant1` | 5 | 0.15 | 0.20 | 0.2501 |
| `67107` | 9 | 0.15 | 0.05 | 0.1523 |
| `grekis` | 2 | 0.15 | 0.30 | 0.1282 |
| `bafcur1` | 9 | 0.15 | 0.05 | 0.0592 |
| `compot1` | 3 | 0.15 | 0.05 | 0.0556 |
| `thlwre1` | 2 | 0.15 | 0.30 | 0.0448 |
| `limpki` | 3 | 0.15 | 0.05 | 0.0049 |

## 4. Proxy Classes

All six proxy classes kept the default proxy weight of **0.05**.

| Label | Positives | Best weight | AP gain |
|---|---:|---:|---:|
| `116570` | 13 | 0.05 | 0.0000 |
| `1491113` | 79 | 0.05 | 0.0000 |
| `1595929` | 0 | 0.05 | N/A |
| `25073` | 12 | 0.05 | 0.0000 |
| `516975` | 13 | 0.05 | 0.0000 |
| `74580` | 0 | 0.05 | N/A |

This suggests proxy mapping helped as a fixed small signal, but the labeled
soundscape calibration did not find a better proxy weight from the current grid.

## 5. Recommendation

The next low-risk variant should be **support-thresholded calibration**:

1. Keep the **0.893** notebook protected.
2. Keep proxy handling unchanged.
3. Apply learned per-class weights only when a class has at least **10**
   labeled soundscape positives.
4. Optionally require an AP gain threshold such as **0.01** before changing a
   default weight.
5. Leave low-support classes at the default blend weight to reduce overfit.

This is safer than a full sequence model and directly follows what the
calibration diagnostics show.

