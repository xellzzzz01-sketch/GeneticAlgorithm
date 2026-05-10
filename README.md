# GA Feature Selection — Fixed Version

Implementasi Genetic Algorithm untuk feature selection pada model LightGBM,
khusus untuk riset material perovskit ganda (5 target: 3 klasifikasi +
2 regresi).

Build sendiri tanpa library GA eksternal (Opytimizer, DEAP, dll).

## File Utama

- **`ga_feature_selection.py`** — script lengkap, siap jalankan di Colab atau lokal.

## Cara Pakai

### Di Google Colab

```python
!git clone https://github.com/xellzzzz01-sketch/GeneticAlgorithm.git
%cd GeneticAlgorithm
!python ga_feature_selection.py
```

Pastikan struktur folder Google Drive kamu seperti ini:

```
MyDrive/Double_Perovskite_Research/
├── Split_90_10/
│   ├── target_is_conductor/Meredig_Magpie_MEGNet/{X_train,X_test,y_train,y_test}.csv
│   ├── target_thermodynamic_stability/...
│   ├── target_is_direct/...
│   ├── target_band_gap/...
│   └── target_formation_energy/...
└── LightGBM_Tuned_Results_90_10/
    └── {target}/Meredig_Magpie_MEGNet/best_params.csv
```

Hasil akan ditulis ke `MyDrive/Double_Perovskite_Research/GA_Results_Fixed/`.

### Di Lokal

```bash
python ga_feature_selection.py
```

Struktur folder sama, tapi di `./Double_Perovskite_Research/`.

## Konfigurasi GA

| Parameter | Nilai | Referensi |
|---|---|---|
| `pop_size` | 50 | Standar |
| `n_generations` | 100 | Standar |
| `crossover_rate` | 0.8 | Goldberg (1989) |
| `mutation_rate` | **1/n_features** dinamis | Back, Hammel, Schwefel (1997) |
| `elite_size` | 2 | Katoch et al. (2021) Sec 3.2.2 |
| `early_stop_gen` | 20 | Xue et al. (2016) |
| `n_folds` | 5 (Stratified/KFold) | Standar |
| `penalty_alpha` | 0.01 | — |

**Selection**: Roulette Wheel Holland (1975) dengan shift + epsilon
(mencegah zero-probability untuk kromosom terburuk).

**Crossover**: Uniform Crossover per-gen (50% swap acak).

**Mutation**: Bit-flip dengan MR = 1/L.

## Ringkasan Fix

Kode ini memperbaiki 14 masalah dari versi-versi sebelumnya:

### Fix Leakage

- **[A]** Imputasi median **per-fold** di CV, bukan global. Sebelumnya,
  `X_train.fillna(global_median)` dipanggil sebelum CV, menyebabkan
  kebocoran antar fold.
- **[B]** Di `train_final_model`, median dihitung dari `X_tr_fit`
  (setelah split val), bukan dari seluruh `X_train`.
- **[C]** SMOTE di final model dilakukan **setelah** split val, sehingga
  synthetic samples tidak masuk ke validation set.

### Fix Stagnasi

- **[D]** Roulette selection pakai shift + epsilon. Versi sebelumnya
  memberikan probabilitas 0 ke kromosom terburuk — menyebabkan selection
  pressure ekstrem dan premature convergence.
- **[E]** R² **tidak** di-clamp ke 0. Nilai negatif dipertahankan agar GA
  bisa membedakan kromosom buruk vs sangat buruk (penting untuk
  early generations di regresi).
- **[F]** Mutation rate **dinamis** = `1/n_features`. Versi sebelumnya
  pakai MR=0.01 tetap, yang terlalu kecil untuk n≥200 dan menyebabkan
  diversity hilang cepat.
- **[G]** `early_stop_gen = 20` (dari 50). Hemat compute ~60% tanpa
  kehilangan kualitas.

### Fix Kualitas Model

- **[H]** Fold yang error mendapat score -1e3, bukan 0 (yang bias
  menguntungkan kromosom bermasalah).
- **[I]** Kalau split val gagal di final model, early stopping
  dimatikan — tidak pakai train sebagai eval_set (yang bikin overfit).
- **[J]** Fitness cache aktif. Elite yang di-copy antar generasi tidak
  dievaluasi ulang. Seed evaluasi deterministik dari hash kromosom.

### Fix v2 (Performance & Robustness)

- **[K]** `n_estimators` dipisah antara **fitness eval** (250, CV cepat)
  dan **final model** (1000, akurasi maksimal). Untuk 25,000 fit ops
  di CV, ini menghemat waktu **3–4x** tanpa kehilangan kualitas seleksi.
- **[L]** Seed dari hash kromosom dijamin `% 2**32` (safe untuk numpy
  seed yang butuh uint32).
- **[M]** Checkpoint validasi `n_features`. Kalau dataset berubah
  antar run, checkpoint dengan shape berbeda otomatis di-discard
  (mencegah `ValueError: shape mismatch` yang tidak jelas).
- **[N]** Safety net 3-tier di final model:
  1. GPU + early stopping
  2. CPU + early stopping
  3. CPU plain (no early stop)
  Kalau ketiga-tiganya gagal, raise RuntimeError yang jelas (bukan
  crash di tengah jalan).

## Output

Untuk tiap target, script menyimpan:

- `metrics_ga_fixed.csv` — metrik test set final
- `ga_chromosome.csv` — kromosom biner terbaik
- `ga_selected_features.csv` — daftar fitur terpilih
- `ga_history.csv` — log per generasi (best, avg, std, n_feat)
- `ga_cv_scores_best.csv` — CV score per fold untuk kromosom terbaik
- `baseline_cv_scores.csv` — baseline (semua fitur)
- `feature_importance_ga.csv` — importance LightGBM
- `predictions_ga_fixed.csv` — prediksi di test set
- `ga_params_config.csv` — semua parameter yang dipakai
- `model_ga_fixed.txt` — model LightGBM (booster format)
- Plot: `ga_convergence.png`, `feature_importance_ga.png`,
  `ga_vs_baseline.png`, `confusion_matrix_ga.png` (klasifikasi),
  `roc_curve_ga.png` (klasifikasi)

Summary global: `GA_Results_Fixed/SUMMARY_GA_FIXED.csv`.

## Checkpoint

Script otomatis menyimpan checkpoint setiap 5 generasi. Kalau eksekusi
terputus (misal Colab disconnect), tinggal jalankan ulang — GA akan
lanjut dari generasi terakhir.

Checkpoint divalidasi terhadap `crossover_rate`, `mutation_rate`,
dan `pop_size`. Kalau parameter berubah, checkpoint di-discard.

## Catatan tentang Wilcoxon Test

Script melakukan uji Wilcoxon antara CV scores GA vs baseline. Dengan
`n_folds=5`, ini hanya menghasilkan **5 paired observations** — statistical
power-nya sangat rendah dan hasil akan hampir selalu `ns` (not significant).

**Untuk TA/publikasi ilmiah**, pertimbangkan:
- Gunakan **repeated stratified K-fold** (misal 5×5 = 25 observations)
  dengan `RepeatedStratifiedKFold` dari sklearn.
- Atau gunakan **nested CV** dengan outer loop sebagai unit observasi.
- Atau laporkan effect size (Cohen's d) selain p-value.

Untuk TA yang sudah cukup rigor tanpa perlu klaim statistik kuat,
melaporkan `Delta_CV` (selisih mean) dan boxplot sudah cukup.

## Referensi

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*.
- Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs*.
- Back, T., Hammel, U., Schwefel, H. P. (1997). Evolutionary computation: Comments on the history and current state. *IEEE TEC*.
- Xue, B., Zhang, M., Browne, W. N., Yao, X. (2016). A survey on evolutionary computation approaches to feature selection. *IEEE TEC*.
- Katoch, S., Chauhan, S. S., Kumar, V. (2021). A review on genetic algorithm: past, present, and future. *Multimedia Tools and Applications*.
- Kusuma, J., et al. (2024). Multi-properties prediction of perovskite materials using machine learning and meta-heuristic feature selection. *Solar Energy*, 286, 113189.
