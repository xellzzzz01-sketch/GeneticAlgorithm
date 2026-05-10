# ============================================================
# GA FEATURE SELECTION — FINAL FIXED VERSION
# Model        : LightGBM
# GA Type      : Elitist Generational GA (build sendiri, bukan Opytimizer)
# Selection    : Roulette Wheel (shift + epsilon, tanpa zero probability)
# Crossover    : Uniform Crossover CR=0.8 (Katoch 2021)
# Mutation     : Bit-flip MR=1/n_features (Back et al. 1997, standar modern)
# SMOTE        : In-fold untuk CV, post-split untuk final model (leakage-free)
# NaN Handling : Eksplisit per kolom, median dari fold train saja
# Backend      : joblib loky (true multiprocessing)
#
# SEMUA FIX DARI AUDIT:
#   [FIX-A] Leakage #1: Hapus global median imputation sebelum CV.
#           Sekarang NaN/Inf dibiarkan masuk ke fitness function,
#           dan imputasi per-fold menggunakan median dari X_tr fold saja.
#
#   [FIX-B] Leakage #2: Di train_final_model, median dihitung dari
#           X_tr_fit (setelah split val), bukan dari seluruh X_train.
#
#   [FIX-C] Leakage #3: SMOTE di final model dilakukan SETELAH split val,
#           sehingga synthetic samples tidak bocor ke X_val_fit.
#
#   [FIX-D] Roulette: Shift-to-min + epsilon kecil, sehingga kromosom
#           terburuk tetap punya probabilitas > 0 (mencegah premature
#           convergence akibat selection pressure ekstrem).
#
#   [FIX-E] R² tidak di-clamp ke 0. Nilai negatif dipertahankan agar
#           GA bisa membedakan kromosom buruk vs sangat buruk
#           (penting untuk early generations di regresi).
#
#   [FIX-F] Mutation rate dinamis: 1.0 / n_features. Untuk n=200
#           menghasilkan ~1 bit flip per kromosom per generasi
#           (Back, Hammel, Schwefel 1997 — standar modern binary GA).
#
#   [FIX-G] early_stop_gen = 20 (dari 50). Konsisten dengan literatur
#           feature selection + komentar awal kode.
#
#   [FIX-H] Fold yang error tidak di-assign score 0.0 (bias sistemik).
#           Ganti dengan nilai sangat buruk (-1e3) agar kromosom yang
#           menyebabkan error tidak diuntungkan.
#
#   [FIX-I] Kalau split val gagal di final model, matikan early stopping
#           (sebelumnya eval_set = train sendiri → overfit).
#
#   [FIX-J] Fitness cache (opsional): elite tidak dievaluasi ulang
#           per generasi. Seed dibuat deterministik dari hash kromosom.
#
# REFERENSI:
#   - Holland (1975)        : Roulette wheel selection
#   - Goldberg (1989)       : Crossover rate tinggi untuk binary GA
#   - Michalewicz (1996)    : Mutasi sebagai operator minor
#   - Back, Hammel, Schwefel (1997) : MR = 1/L untuk binary
#   - Katoch et al. (2021)  : Elitism, uniform crossover structure
#   - Xue et al. (2016)     : Feature selection via evolutionary computation
#   - Kusuma et al. (2024)  : Framework GA + LightGBM untuk perovskite
#
# PARAMETER GA:
#   pop_size=50, n_gen=100, CR=0.8, MR=1/n_features (dinamis)
#   elite=2, early_stop=20, penalty_alpha=0.01
# ============================================================

# ══════════════════════════════════════════
# [0] SETUP
# ══════════════════════════════════════════
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import os
import warnings
import random
import subprocess
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
try:
    from google.colab import drive as _chk  # noqa
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def pip_install(pkg):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


for pkg in ['lightgbm', 'scikit-learn', 'imbalanced-learn', 'joblib', 'scipy']:
    pip_install(pkg)

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import (
    StratifiedKFold, KFold,
    train_test_split as sk_split,
)
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon
from joblib import Parallel, delayed

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

lgb_version  = tuple(int(x) for x in lgb.__version__.split('.')[:2])
GOSS_NEW_API = lgb_version >= (4, 0)

print("=" * 65)
print("  GA FEATURE SELECTION — FINAL FIXED")
print("  Model  : LightGBM")
print("  GA     : Elitist Generational (build sendiri)")
print("  Select : Roulette Wheel (shift + epsilon)")
print("  CR=0.8 | MR=1/n_features | Elite=2 | EarlyStop=20")
print("  Leakage-free: imputasi per-fold + SMOTE post-split")
print("=" * 65)


# ══════════════════════════════════════════
# [A] PATH
# ══════════════════════════════════════════
if IN_COLAB:
    BASE_FOLDER = '/content/drive/MyDrive/Double_Perovskite_Research'
    CKPT_DIR    = '/content'
else:
    BASE_FOLDER = 'Double_Perovskite_Research'
    CKPT_DIR    = '.'

SPLIT_FOLDER  = os.path.join(BASE_FOLDER, 'Split_90_10')
TUNED_FOLDER  = os.path.join(BASE_FOLDER, 'LightGBM_Tuned_Results_90_10')
RESULT_FOLDER = os.path.join(BASE_FOLDER, 'GA_Results_Fixed')
os.makedirs(RESULT_FOLDER, exist_ok=True)

CKPT_LOCATIONS = [CKPT_DIR, os.getcwd(), RESULT_FOLDER]

print(f"  Split  : {SPLIT_FOLDER}")
print(f"  Tuned  : {TUNED_FOLDER}")
print(f"  Output : {RESULT_FOLDER}")

if not os.path.exists(SPLIT_FOLDER):
    raise FileNotFoundError(f"SPLIT_FOLDER tidak ada: {SPLIT_FOLDER}")


# ══════════════════════════════════════════
# [B] TARGET CONFIG
# ══════════════════════════════════════════
TARGET_CONFIG = {
    'is_conductor': {
        'type'          : 'classification',
        'folder'        : 'target_is_conductor',
        'metric_main'   : 'F1_weighted',
        'use_smote'     : False,
        'desc'          : 'Conductor vs Non-Conductor',
        'class_names'   : ['Non-Conductor', 'Conductor'],
        'minority_label': 1,
    },
    'thermodynamic_stability': {
        'type'          : 'classification',
        'folder'        : 'target_thermodynamic_stability',
        'metric_main'   : 'F1_weighted',
        'use_smote'     : True,
        'desc'          : 'Thermodynamic Stability',
        'class_names'   : ['Unstable', 'Stable'],
        'minority_label': 1,
    },
    'is_direct': {
        'type'          : 'classification',
        'folder'        : 'target_is_direct',
        'metric_main'   : 'F1_weighted',
        'use_smote'     : True,
        'desc'          : 'Direct vs Indirect Band Gap',
        'class_names'   : ['Indirect', 'Direct'],
        'minority_label': 1,
    },
    'band_gap': {
        'type'          : 'regression',
        'folder'        : 'target_band_gap',
        'metric_main'   : 'R2',
        'use_smote'     : False,
        'desc'          : 'Band Gap Eg (eV)',
        'class_names'   : None,
        'minority_label': None,
    },
    'formation_energy': {
        'type'          : 'regression',
        'folder'        : 'target_formation_energy',
        'metric_main'   : 'R2',
        'use_smote'     : False,
        'desc'          : 'Formation Energy (eV/atom)',
        'class_names'   : None,
        'minority_label': None,
    },
}

METHOD = 'Meredig_Magpie_MEGNet'


# ══════════════════════════════════════════
# [C] PARAMETER GA
#
# [FIX-F] mutation_rate dihitung dinamis di run_ga() sebagai
#         1.0 / n_features (Back et al. 1997) — sekitar 1 flip
#         per kromosom per generasi. Lihat compute_mutation_rate().
#
# [FIX-G] early_stop_gen = 20 (turun dari 50) — sesuai literatur
#         feature selection (Xue et al. 2016) dan komentar awal.
# ══════════════════════════════════════════
GA_PARAMS = {
    'pop_size'       : 50,
    'n_generations'  : 100,
    'crossover_rate' : 0.8,
    # mutation_rate TIDAK diset di sini — dihitung dinamis per target
    'elite_size'     : 2,
    'n_folds'        : 5,
    'early_stop_gen' : 20,      # [FIX-G]
    'penalty_alpha'  : 0.01,
    'ckpt_interval'  : 5,
    'n_jobs_parallel': -1,
    'use_fitness_cache': True,  # [FIX-J]
}

BG_COLOR    = '#0D1117'
PANEL_COLOR = '#161B22'


def compute_mutation_rate(n_features):
    """
    [FIX-F] MR = 1/L (Back, Hammel, Schwefel 1997).
    Standar modern untuk binary GA. Menjamin rata-rata
    1 bit flip per kromosom per generasi.
    """
    return max(1.0 / n_features, 1e-4)


# ══════════════════════════════════════════
# [D] DETEKSI GPU — HANYA UNTUK MODEL FINAL
# ══════════════════════════════════════════
def _detect_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode != 0:
            return 'cpu', 'nvidia-smi tidak ditemukan'
    except Exception:
        return 'cpu', 'nvidia-smi tidak tersedia'
    try:
        _X = np.random.rand(50, 5)
        _y = (np.random.rand(50) > 0.5).astype(int)
        _m = lgb.LGBMClassifier(
            n_estimators=5, verbose=-1,
            device='gpu', random_state=42, n_jobs=1)
        _m.fit(_X, _y)
        return 'gpu', 'GPU OK untuk model final'
    except Exception as e:
        return 'cpu', f'GPU gagal ({e})'


LGB_DEVICE_FINAL, DEVICE_MSG = _detect_gpu()
LGB_DEVICE_FITNESS = 'cpu'   # GPU tidak thread-safe untuk paralel

print(f"\n  LightGBM : {lgb.__version__}  |  GOSS baru: {GOSS_NEW_API}")
print(f"  Device fitness eval : CPU (paralel, aman)")
print(f"  Device model final  : {LGB_DEVICE_FINAL.upper()} ({DEVICE_MSG})")
print("  Library siap\n")


# ══════════════════════════════════════════
# [E] LOAD TUNED PARAMS
# ══════════════════════════════════════════
PARAM_TYPES = {
    'num_leaves'       : int,
    'max_depth'        : int,
    'min_child_samples': int,
    'max_bin'          : int,
    'learning_rate'    : float,
    'colsample_bytree' : float,
    'subsample'        : float,
    'reg_alpha'        : float,
    'reg_lambda'       : float,
    'top_rate'         : float,
    'other_rate'       : float,
}

DEFAULT_PARAMS = {
    'num_leaves'       : 31,
    'max_depth'        : -1,
    'min_child_samples': 20,
    'learning_rate'    : 0.05,
    'colsample_bytree' : 0.8,
    'subsample'        : 0.8,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 0.1,
    'top_rate'         : 0.05,
    'other_rate'       : 0.05,
}


def load_tuned_params(target_name, method):
    csv_path = os.path.join(
        TUNED_FOLDER, target_name, method, 'best_params.csv')
    if not os.path.exists(csv_path):
        print(f"         PERINGATAN: best_params.csv tidak ada -> pakai DEFAULT")
        return DEFAULT_PARAMS.copy(), False
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        if 'param' not in df.columns or 'value' not in df.columns:
            raise ValueError(f"Kolom tidak sesuai: {df.columns.tolist()}")
        params = {}
        for _, row in df.iterrows():
            key = str(row['param']).strip()
            val = row['value']
            if key in PARAM_TYPES:
                params[key] = PARAM_TYPES[key](val)
        print(f"         OK params dimuat: {csv_path}")
        return params, True
    except Exception as e:
        print(f"         PERINGATAN: Error baca params ({e}) -> pakai DEFAULT")
        return DEFAULT_PARAMS.copy(), False


def build_lgb_params(tuned_params, task_type, device='cpu', n_jobs=1):
    top_rate   = float(tuned_params.get('top_rate',   0.05))
    other_rate = float(tuned_params.get('other_rate', 0.05))
    if top_rate + other_rate >= 1.0:
        top_rate, other_rate = 0.05, 0.05

    params = {
        'num_leaves'       : int(tuned_params.get('num_leaves', 31)),
        'max_depth'        : int(tuned_params.get('max_depth', -1)),
        'min_child_samples': int(tuned_params.get('min_child_samples', 20)),
        'learning_rate'    : float(tuned_params.get('learning_rate', 0.05)),
        'colsample_bytree' : float(tuned_params.get('colsample_bytree', 0.8)),
        'reg_alpha'        : float(tuned_params.get('reg_alpha', 0.1)),
        'reg_lambda'       : float(tuned_params.get('reg_lambda', 0.1)),
        'n_estimators'     : 1000,   # + early stopping
        'n_jobs'           : n_jobs,
        'random_state'     : GLOBAL_SEED,
        'verbose'          : -1,
        'device'           : device,
    }

    if 'max_bin' in tuned_params:
        params['max_bin'] = int(tuned_params['max_bin'])
    if 'subsample' in tuned_params:
        params['subsample'] = float(tuned_params['subsample'])

    if GOSS_NEW_API:
        params['boosting_type']        = 'gbdt'
        params['data_sample_strategy'] = 'goss'
        params['top_rate']             = top_rate
        params['other_rate']           = other_rate
    else:
        params['boosting_type'] = 'goss'
        params['top_rate']      = top_rate
        params['other_rate']    = other_rate

    if task_type == 'classification':
        params['objective']    = 'binary'
        params['metric']       = 'binary_logloss'
        params['class_weight'] = 'balanced'
    else:
        params['objective'] = 'regression'
        params['metric']    = 'rmse'

    return params


# ══════════════════════════════════════════
# [F] LOAD DATA
# ══════════════════════════════════════════
def load_split_data(target_folder_path, method):
    method_path = os.path.join(target_folder_path, method)
    if not os.path.exists(method_path):
        print(f"         ERROR: Folder tidak ada: {method_path}")
        return None, None, None, None
    try:
        X_train = pd.read_csv(os.path.join(method_path, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(method_path, 'y_train.csv')).squeeze()
        X_test  = pd.read_csv(os.path.join(method_path, 'X_test.csv'))
        y_test  = pd.read_csv(os.path.join(method_path, 'y_test.csv')).squeeze()
        print(f"         Train: {X_train.shape}  |  Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"         ERROR load data: {e}")
        return None, None, None, None


# ══════════════════════════════════════════
# [G] EVALUASI METRIK
# ══════════════════════════════════════════
def evaluate_classification(y_true, y_pred, y_prob, minority_label=1):
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_ma = f1_score(y_true, y_pred, average='macro', zero_division=0)
    labels = np.unique(y_true)
    f1_mi  = (f1_score(y_true, y_pred, labels=[minority_label],
                        average='micro', zero_division=0)
              if minority_label in labels else float('nan'))
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_val = float('nan')
    return {
        'Accuracy'   : round(acc,     4),
        'Precision'  : round(prec,    4),
        'Recall'     : round(rec,     4),
        'F1_weighted': round(f1_w,    4),
        'F1_macro'   : round(f1_ma,   4),
        'F1_minority': round(f1_mi,   4) if not np.isnan(f1_mi) else float('nan'),
        'AUC'        : round(auc_val, 4) if not np.isnan(auc_val) else float('nan'),
    }


def evaluate_regression(y_true, y_pred):
    return {
        'MAE' : round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        'R2'  : round(r2_score(y_true, y_pred), 4),
    }


# ══════════════════════════════════════════
# [H] NaN IMPUTATION — EKSPLISIT PER KOLOM
# ══════════════════════════════════════════
def impute_nan(X, col_medians):
    """
    Imputasi NaN/Inf eksplisit per kolom.
    col_medians WAJIB dari fold train saja (leakage-free).
    """
    X_clean = X.copy().astype(np.float64)
    for j in range(X_clean.shape[1]):
        bad = ~np.isfinite(X_clean[:, j])
        if bad.any():
            fill = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
            X_clean[bad, j] = fill
    return X_clean


# ══════════════════════════════════════════
# [I] FITNESS FUNCTION — SATU KROMOSOM
#
# [FIX-A] X_arr yang masuk BOLEH mengandung NaN/Inf.
#         Imputasi dilakukan di dalam fold, pakai median
#         dari X_tr fold SAJA. Ini leakage-free antar fold.
#
# [FIX-E] R² tidak di-clamp. Nilai negatif dipertahankan
#         agar GA bisa membedakan kromosom buruk vs sangat buruk.
#
# [FIX-H] Fold yang error return -1e3 (bukan 0.0) agar
#         kromosom yang bermasalah tidak diuntungkan.
# ══════════════════════════════════════════
FOLD_ERROR_SCORE = -1e3          # [FIX-H]
EMPTY_CHROM_FITNESS = -1e6       # fitness sangat buruk untuk kromosom kosong


def _evaluate_one_chromosome(chromosome, X_arr, y_arr,
                              task_type, lgb_params,
                              n_folds, penalty_alpha,
                              use_smote, seed):
    selected_idx = np.where(chromosome == 1)[0]
    n_total      = len(chromosome)
    n_selected   = len(selected_idx)

    if n_selected == 0:
        return EMPTY_CHROM_FITNESS, []

    # Guard: pastikan indeks dalam batas kolom
    selected_idx = selected_idx[selected_idx < X_arr.shape[1]]
    if len(selected_idx) == 0:
        return EMPTY_CHROM_FITNESS, []

    X_sel = X_arr[:, selected_idx]

    if task_type == 'classification':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scores = []

    for tr_idx, val_idx in kf.split(X_sel, y_arr):
        X_tr  = X_sel[tr_idx].copy()
        X_val = X_sel[val_idx].copy()
        y_tr  = y_arr[tr_idx].copy()
        y_val = y_arr[val_idx].copy()

        # [FIX-A] Median hanya dari fold training (leakage-free)
        col_med = np.nanmedian(X_tr, axis=0)
        X_tr    = impute_nan(X_tr,  col_med)
        X_val   = impute_nan(X_val, col_med)

        # SMOTE hanya pada training fold (leakage-free)
        if use_smote and task_type == 'classification':
            classes, counts = np.unique(y_tr, return_counts=True)
            if len(classes) >= 2 and counts.min() >= 2:
                k_n = max(1, min(5, counts.min() - 1))
                try:
                    sm = SMOTE(random_state=seed, k_neighbors=k_n)
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except Exception:
                    pass

        if task_type == 'classification':
            model = lgb.LGBMClassifier(**lgb_params)
        else:
            model = lgb.LGBMRegressor(**lgb_params)

        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
            )
        except Exception:
            try:
                model.fit(X_tr, y_tr)
            except Exception:
                scores.append(FOLD_ERROR_SCORE)    # [FIX-H]
                continue

        y_pred = model.predict(X_val)

        if task_type == 'classification':
            score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        else:
            # [FIX-E] TIDAK di-clamp — nilai negatif dipertahankan
            score = r2_score(y_val, y_pred)

        scores.append(score)

    if not scores:
        return EMPTY_CHROM_FITNESS, []

    base_score = float(np.mean(scores))
    penalty    = penalty_alpha * (n_selected / n_total)
    fitness    = base_score - penalty

    return fitness, scores


# ══════════════════════════════════════════
# [I2] FITNESS CACHE (FIX-J)
# ══════════════════════════════════════════
class FitnessCache:
    """
    Cache hasil fitness berdasarkan hash kromosom.
    Menghemat compute untuk elite yang di-copy antar generasi.
    Seed evaluasi deterministik dari hash kromosom agar
    hasil sama walaupun kromosom muncul di posisi berbeda.
    """
    def __init__(self, max_size=500):
        self.cache = {}
        self.max_size = max_size

    @staticmethod
    def key(chrom):
        return chrom.tobytes()

    @staticmethod
    def seed_from_chrom(chrom, base_seed):
        # Seed deterministik dari 4 byte pertama hash kromosom
        return int(base_seed + int.from_bytes(chrom.tobytes()[:4], 'little') % 100000)

    def get(self, chrom):
        return self.cache.get(self.key(chrom))

    def set(self, chrom, fitness, cv_scores):
        if len(self.cache) >= self.max_size:
            # FIFO eviction sederhana
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[self.key(chrom)] = (fitness, list(cv_scores))

    def clear(self):
        self.cache.clear()


def fitness_population_parallel(population, X_arr, y_arr,
                                 task_type, lgb_params,
                                 n_folds, penalty_alpha,
                                 use_smote, n_jobs, base_seed,
                                 cache=None):
    """
    Evaluasi populasi secara paralel dengan optional cache.
    [FIX-J] Elite tidak di-evaluasi ulang kalau cache aktif.
    """
    n = len(population)
    results = [None] * n
    to_eval_idx = []
    to_eval = []

    if cache is not None:
        for i, c in enumerate(population):
            cached = cache.get(c)
            if cached is not None:
                results[i] = cached
            else:
                to_eval_idx.append(i)
                to_eval.append(c)
    else:
        to_eval_idx = list(range(n))
        to_eval = population

    if to_eval:
        new_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_evaluate_one_chromosome)(
                c, X_arr, y_arr, task_type,
                lgb_params, n_folds, penalty_alpha,
                use_smote,
                FitnessCache.seed_from_chrom(c, base_seed),  # seed dari hash
            )
            for c in to_eval
        )
        for idx, c, res in zip(to_eval_idx, to_eval, new_results):
            results[idx] = res
            if cache is not None:
                cache.set(c, res[0], res[1])

    fitness_scores = np.array([r[0] for r in results])
    cv_scores_pop  = [r[1] for r in results]
    return fitness_scores, cv_scores_pop


# ══════════════════════════════════════════
# [I3] BASELINE — EVALUASI SEMUA FITUR
# ══════════════════════════════════════════
def run_baseline(X_arr, y_arr, task_type, lgb_params,
                 n_folds=5, use_smote=False):
    print(f"         Baseline (semua fitur={X_arr.shape[1]}) "
          f"-- CV {n_folds}-fold...", flush=True)
    all_chrom = np.ones(X_arr.shape[1], dtype=int)
    fitness, cv_scores = _evaluate_one_chromosome(
        all_chrom, X_arr, y_arr, task_type,
        lgb_params, n_folds,
        penalty_alpha=0.0,
        use_smote=use_smote,
        seed=GLOBAL_SEED,
    )
    cv_mean = float(np.mean(cv_scores)) if cv_scores else float('nan')
    cv_std  = float(np.std(cv_scores))  if cv_scores else float('nan')
    print(f"         Baseline CV: {cv_mean:.5f} +/- {cv_std:.5f}")
    return list(cv_scores), cv_mean, cv_std


# ══════════════════════════════════════════
# [J] OPERATOR GA
# ══════════════════════════════════════════

# ── [J1] INISIALISASI POPULASI ────────────
def init_population(pop_size, n_features, seed=42):
    rng      = np.random.RandomState(seed)
    min_feat = max(2, int(0.01 * n_features))
    pop      = []
    for _ in range(pop_size):
        chrom = (rng.random(n_features) < 0.5).astype(int)
        if chrom.sum() < min_feat:
            zero_idx = np.where(chrom == 0)[0]
            chosen   = rng.choice(
                zero_idx,
                size=min_feat - int(chrom.sum()),
                replace=False,
            )
            chrom[chosen] = 1
        pop.append(chrom)
    return pop


# ── [J2] ROULETTE WHEEL SELECTION (FIX-D) ──
def roulette_selection(population, fitness_scores, rng):
    """
    [FIX-D] Roulette Wheel dengan shift + epsilon.
    Mencegah probabilitas 0 untuk kromosom terburuk yang
    membuat selection pressure ekstrem dan premature convergence.

    p_i = (f_i - f_min + eps) / sum(f_i - f_min + eps)
    """
    scores = np.array(fitness_scores, dtype=float)
    # Shift ke non-negatif + epsilon agar worst tetap > 0
    scores = scores - scores.min() + 1e-6
    total  = scores.sum()
    n      = len(population)

    if total <= 0 or not np.isfinite(total):
        idx = rng.randint(0, n)
    else:
        probs = scores / total
        idx   = rng.choice(n, p=probs)

    return population[idx].copy()


# ── [J3] UNIFORM CROSSOVER ────────────────
def uniform_crossover(p1, p2, crossover_rate, rng):
    """
    Uniform Crossover (Katoch et al. 2021 Sec 3.2.3).
    CR=0.8: 80% pasangan mengalami rekombinasi bit-level acak.
    """
    if rng.random() < crossover_rate:
        mask = rng.random(len(p1)) < 0.5
        c1   = np.where(mask, p1, p2)
        c2   = np.where(mask, p2, p1)
        return c1, c2
    return p1.copy(), p2.copy()


# ── [J4] BIT-FLIP MUTATION ────────────────
def bit_flip_mutation(chrom, mutation_rate, rng):
    """
    Bit-flip Mutation (Michalewicz 1996, Back et al. 1997).
    MR = 1/L -> rata-rata 1 bit flip per kromosom per generasi.
    """
    min_feat  = max(2, int(0.01 * len(chrom)))
    c         = chrom.copy()
    flip_mask = rng.random(len(c)) < mutation_rate
    c         = np.where(flip_mask, 1 - c, c)

    # Guard minimum fitur
    if c.sum() < min_feat:
        zero_idx = np.where(c == 0)[0]
        if len(zero_idx) > 0:
            needed = min(min_feat - int(c.sum()), len(zero_idx))
            chosen = rng.choice(zero_idx, size=needed, replace=False)
            c[chosen] = 1

    return c


# ══════════════════════════════════════════
# [K] CHECKPOINT
# ══════════════════════════════════════════
def _ckpt_filename(target_name):
    return f'ga_fixed_ckpt_{target_name}_{METHOD}.json'


def _all_ckpt_dirs(save_dir):
    seen, result = set(), []
    for d in [save_dir] + CKPT_LOCATIONS:
        key = os.path.realpath(d)
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def _validate_ckpt(ckpt, n_features):
    errors = []
    expected_mr = compute_mutation_rate(n_features)
    for key, cfg_val in [
        ('crossover_rate', GA_PARAMS['crossover_rate']),
        ('mutation_rate',  expected_mr),
        ('pop_size',       GA_PARAMS['pop_size']),
    ]:
        ckpt_val = ckpt.get(key)
        if ckpt_val is not None:
            if abs(float(ckpt_val) - cfg_val) > 1e-6:
                errors.append(f"{key}: ckpt={ckpt_val} != config={cfg_val}")
    return errors


def load_checkpoint(save_dir, target_name, n_features):
    fname      = _ckpt_filename(target_name)
    candidates = []
    for src in _all_ckpt_dirs(save_dir):
        path = os.path.join(src, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r') as f:
                ckpt = json.load(f)
            errors = _validate_ckpt(ckpt, n_features)
            if errors:
                for e in errors:
                    print(f"         CKPT tidak valid: {e}")
                print(f"         -> Checkpoint diabaikan")
                continue
            candidates.append((ckpt['last_gen'], path, ckpt))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_path, best_ckpt = candidates[0]
    print(f"         Checkpoint: {best_path} (gen={best_ckpt['last_gen']})")
    return best_ckpt


def save_checkpoint(ckpt_data, save_dir, target_name):
    fname = _ckpt_filename(target_name)
    for dest in _all_ckpt_dirs(save_dir):
        try:
            os.makedirs(dest, exist_ok=True)
            path = os.path.join(dest, fname)
            tmp  = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(ckpt_data, f)
            os.replace(tmp, path)
        except Exception as e:
            print(f"         PERINGATAN simpan ckpt: {e}")


def delete_checkpoint(save_dir, target_name):
    fname = _ckpt_filename(target_name)
    for dest in _all_ckpt_dirs(save_dir):
        path = os.path.join(dest, fname)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


# ══════════════════════════════════════════
# [L] MAIN GA LOOP — ELITIST GENERATIONAL
# ══════════════════════════════════════════
def run_ga(X_arr, y_arr, task_type, feature_names,
           lgb_params, target_name='', use_smote=False):

    pop_size       = GA_PARAMS['pop_size']
    n_generations  = GA_PARAMS['n_generations']
    crossover_rate = GA_PARAMS['crossover_rate']
    elite_size     = GA_PARAMS['elite_size']
    n_folds        = GA_PARAMS['n_folds']
    early_stop_gen = GA_PARAMS['early_stop_gen']
    penalty_alpha  = GA_PARAMS['penalty_alpha']
    ckpt_interval  = GA_PARAMS['ckpt_interval']
    n_jobs         = GA_PARAMS['n_jobs_parallel']
    use_cache      = GA_PARAMS['use_fitness_cache']

    n_features = X_arr.shape[1]

    # [FIX-F] Mutation rate dinamis
    mutation_rate = compute_mutation_rate(n_features)

    save_dir = os.path.join(RESULT_FOLDER, target_name, METHOD)
    os.makedirs(save_dir, exist_ok=True)

    expected_flips = mutation_rate * n_features

    print(f"\n         GA Config — Elitist Generational GA")
    print(f"         {'-'*55}")
    print(f"         Target     : {target_name} | {METHOD}")
    print(f"         N fitur    : {n_features}")
    print(f"         Pop={pop_size}  Gen={n_generations}  "
          f"CR={crossover_rate}  MR={mutation_rate:.5f} (=1/L)")
    print(f"         Elite={elite_size}  EarlyStop={early_stop_gen}")
    print(f"         SMOTE      : {'Ya (in-fold)' if use_smote else 'Tidak'}")
    print(f"         n_jobs     : {n_jobs} (backend=loky)")
    print(f"         Cache      : {'AKTIF' if use_cache else 'MATI'}")
    print(f"         Expected   : ~{expected_flips:.1f} bit flip/kromosom/gen")
    print(f"         [FIX-A..J] Leakage-free + Anti-stagnasi")
    print(f"         {'-'*55}")

    cache = FitnessCache(max_size=500) if use_cache else None

    ckpt = load_checkpoint(save_dir, target_name, n_features)

    if ckpt is not None:
        population      = [np.array(p) for p in ckpt['population']]
        best_chromosome = np.array(ckpt['best_chromosome'])
        best_fitness    = ckpt['best_fitness']
        no_improve_cnt  = ckpt['no_improve_cnt']
        history         = ckpt['history']
        gen_stats       = ckpt['gen_stats']
        start_gen       = ckpt['last_gen']
        best_cv_scores  = ckpt.get('best_cv_scores', [])
        if no_improve_cnt >= early_stop_gen:
            print(f"         INFO: Sudah early stop -> skip GA")
            selected_idx   = np.where(best_chromosome == 1)[0].tolist()
            selected_names = [feature_names[i] for i in selected_idx]
            return (best_chromosome, selected_idx, selected_names,
                    history, gen_stats, best_cv_scores)
    else:
        population      = init_population(pop_size, n_features, seed=GLOBAL_SEED)
        best_chromosome = None
        best_fitness    = -np.inf
        no_improve_cnt  = 0
        history         = []
        gen_stats       = []
        start_gen       = 0
        best_cv_scores  = []

    print(f"\n         {'Gen':>5} | {'BestFit':>8} | {'AvgFit':>8} | "
          f"{'StdFit':>7} | {'N_Feat':>7} | {'Waktu':>7} | Status")
    print(f"         {'-'*80}")

    for gen in range(start_gen, n_generations):
        t0      = time.time()
        gen_rng = np.random.RandomState(GLOBAL_SEED + gen * 1000)

        # 1. Evaluasi seluruh populasi
        fitness_scores, cv_scores_pop = fitness_population_parallel(
            population, X_arr, y_arr, task_type,
            lgb_params, n_folds, penalty_alpha,
            use_smote, n_jobs=n_jobs,
            base_seed=GLOBAL_SEED,
            cache=cache,
        )

        # 2. Catat best
        gen_best_idx = int(np.argmax(fitness_scores))
        gen_best_fit = float(fitness_scores[gen_best_idx])
        gen_avg_fit  = float(np.mean(fitness_scores))
        gen_std_fit  = float(np.std(fitness_scores))
        gen_n_feat   = int(population[gen_best_idx].sum())

        improved = gen_best_fit > best_fitness
        if improved:
            best_fitness    = gen_best_fit
            best_chromosome = population[gen_best_idx].copy()
            best_cv_scores  = cv_scores_pop[gen_best_idx]
            no_improve_cnt  = 0
        else:
            no_improve_cnt += 1

        elapsed = time.time() - t0
        history.append(best_fitness)
        gen_stats.append({
            'generation'  : gen + 1,
            'best_fitness': round(gen_best_fit, 5),
            'avg_fitness' : round(gen_avg_fit,  5),
            'std_fitness' : round(gen_std_fit,  5),
            'best_global' : round(best_fitness, 5),
            'n_features'  : gen_n_feat,
            'no_improve'  : no_improve_cnt,
            'improved'    : improved,
            'elapsed_sec' : round(elapsed, 2),
        })

        if (gen + 1) % 5 == 0 or gen == start_gen or improved:
            flag = 'BARU' if improved else f'({no_improve_cnt}x stagnan)'
            print(f"         {gen+1:>5} | {gen_best_fit:>8.5f} | "
                  f"{gen_avg_fit:>8.5f} | {gen_std_fit:>7.5f} | "
                  f"{gen_n_feat:>7d} | {elapsed:>6.1f}s | {flag}")

        # Checkpoint
        if (gen + 1) % ckpt_interval == 0:
            save_checkpoint({
                'target_name'    : target_name,
                'method'         : METHOD,
                'crossover_rate' : crossover_rate,
                'mutation_rate'  : mutation_rate,
                'pop_size'       : pop_size,
                'last_gen'       : gen + 1,
                'best_fitness'   : best_fitness,
                'no_improve_cnt' : no_improve_cnt,
                'history'        : history,
                'gen_stats'      : gen_stats,
                'best_chromosome': best_chromosome.tolist() if best_chromosome is not None else [],
                'population'     : [p.tolist() for p in population],
                'best_cv_scores' : best_cv_scores,
            }, save_dir, target_name)

        # Early stopping
        if no_improve_cnt >= early_stop_gen:
            print(f"\n         Early stop gen {gen+1} "
                  f"(stagnan {early_stop_gen} generasi)")
            save_checkpoint({
                'target_name'    : target_name,
                'method'         : METHOD,
                'crossover_rate' : crossover_rate,
                'mutation_rate'  : mutation_rate,
                'pop_size'       : pop_size,
                'last_gen'       : gen + 1,
                'best_fitness'   : best_fitness,
                'no_improve_cnt' : no_improve_cnt,
                'history'        : history,
                'gen_stats'      : gen_stats,
                'best_chromosome': best_chromosome.tolist() if best_chromosome is not None else [],
                'population'     : [p.tolist() for p in population],
                'best_cv_scores' : best_cv_scores,
            }, save_dir, target_name)
            break

        # 3. Elitism — copy best elite_size ke populasi baru
        sorted_idx = np.argsort(fitness_scores)[::-1]
        new_pop    = [population[i].copy() for i in sorted_idx[:elite_size]]

        # 4. Selection -> Crossover -> Mutation
        while len(new_pop) < pop_size:
            p1 = roulette_selection(population, fitness_scores, gen_rng)
            p2 = roulette_selection(population, fitness_scores, gen_rng)
            c1, c2 = uniform_crossover(p1, p2, crossover_rate, gen_rng)
            c1 = bit_flip_mutation(c1, mutation_rate, gen_rng)
            c2 = bit_flip_mutation(c2, mutation_rate, gen_rng)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop[:pop_size]

    if best_chromosome is None:
        # Safety net: kalau entah bagaimana tidak ada best ditemukan
        best_chromosome = population[0].copy()

    selected_idx   = np.where(best_chromosome == 1)[0].tolist()
    selected_names = [feature_names[i] for i in selected_idx]
    reduction_pct  = (1 - len(selected_idx) / n_features) * 100

    print(f"\n         -- GA Selesai --")
    print(f"            Best fitness   : {best_fitness:.5f}")
    print(f"            Fitur terpilih : {len(selected_idx)} / {n_features} "
          f"(reduksi {reduction_pct:.1f}%)")

    return (best_chromosome, selected_idx, selected_names,
            history, gen_stats, best_cv_scores)


# ══════════════════════════════════════════
# [M] MODEL FINAL — DENGAN FIX LEAKAGE
#
# [FIX-B] Median dari X_tr_fit_raw saja (setelah split val),
#         bukan dari seluruh X_train.
#
# [FIX-C] SMOTE SETELAH split val, hanya di X_tr_fit
#         (bukan di X_val_fit yang dipakai untuk early stopping).
#
# [FIX-I] Kalau split val gagal, matikan early stopping
#         (jangan pakai train sebagai eval_set).
# ══════════════════════════════════════════
def train_final_model(X_train_df, X_test_df, y_train, y_test,
                      selected_idx, task_type, cfg,
                      tuned_params, use_smote=False):

    # Guard: validasi batas indeks
    n_cols_tr = X_train_df.shape[1]
    n_cols_te = X_test_df.shape[1]
    valid_idx = [i for i in selected_idx
                 if i < n_cols_tr and i < n_cols_te]
    if len(valid_idx) < len(selected_idx):
        dropped = len(selected_idx) - len(valid_idx)
        print(f"         PERINGATAN: {dropped} indeks di luar batas -> dibuang")
    if not valid_idx:
        raise ValueError("Tidak ada fitur valid untuk model final!")

    X_tr_raw = X_train_df.iloc[:, valid_idx].values.astype(np.float64)
    X_te_raw = X_test_df.iloc[:, valid_idx].values.astype(np.float64)
    y_tr     = np.array(y_train).ravel()
    y_te     = np.array(y_test).ravel()

    # [FIX-B] Split val DULU (sebelum imputasi + SMOTE)
    val_size = min(0.1, max(0.05, 50 / len(X_tr_raw)))
    use_early_stopping = True
    try:
        stratify = y_tr if task_type == 'classification' else None
        X_tr_fit_raw, X_val_fit_raw, y_tr_fit, y_val_fit = sk_split(
            X_tr_raw, y_tr, test_size=val_size,
            random_state=GLOBAL_SEED, stratify=stratify,
        )
    except ValueError:
        try:
            X_tr_fit_raw, X_val_fit_raw, y_tr_fit, y_val_fit = sk_split(
                X_tr_raw, y_tr, test_size=val_size,
                random_state=GLOBAL_SEED,
            )
        except Exception:
            # [FIX-I] Matikan early stopping jika split gagal
            print("         PERINGATAN: split val gagal -> tanpa early stopping")
            X_tr_fit_raw = X_tr_raw
            y_tr_fit     = y_tr
            X_val_fit_raw = None
            y_val_fit     = None
            use_early_stopping = False

    # [FIX-B] Median dari X_tr_fit_raw saja
    col_med = np.nanmedian(X_tr_fit_raw, axis=0)
    X_tr_fit = impute_nan(X_tr_fit_raw, col_med)
    if X_val_fit_raw is not None:
        X_val_fit = impute_nan(X_val_fit_raw, col_med)
    else:
        X_val_fit = None
    X_te = impute_nan(X_te_raw, col_med)

    # [FIX-C] SMOTE SETELAH split, hanya di X_tr_fit
    if use_smote and task_type == 'classification':
        classes, counts = np.unique(y_tr_fit, return_counts=True)
        if len(classes) >= 2 and counts.min() >= 2:
            k_n = max(1, min(5, counts.min() - 1))
            try:
                sm = SMOTE(random_state=GLOBAL_SEED, k_neighbors=k_n)
                X_tr_fit, y_tr_fit = sm.fit_resample(X_tr_fit, y_tr_fit)
                print(f"         SMOTE final (post-split): {X_tr_fit.shape[0]} sampel")
            except Exception as e:
                print(f"         SMOTE gagal ({e}) -- lanjut tanpa")

    # Build params untuk model final (GPU + n_jobs=-1)
    final_params = build_lgb_params(
        tuned_params, task_type,
        device=LGB_DEVICE_FINAL, n_jobs=-1,
    )
    final_params['n_estimators'] = 1000

    if task_type == 'classification':
        model = lgb.LGBMClassifier(**final_params)
    else:
        model = lgb.LGBMRegressor(**final_params)

    fit_ok = False
    if use_early_stopping and X_val_fit is not None:
        try:
            model.fit(
                X_tr_fit, y_tr_fit,
                eval_set=[(X_val_fit, y_val_fit)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            fit_ok = True
        except Exception as e:
            print(f"         GPU/early-stop error ({e}) -> fallback CPU")

    if not fit_ok:
        # Fallback: CPU + tanpa early stopping (aman)
        final_params['device'] = 'cpu'
        model = (lgb.LGBMClassifier(**final_params)
                 if task_type == 'classification'
                 else lgb.LGBMRegressor(**final_params))
        if use_early_stopping and X_val_fit is not None:
            model.fit(
                X_tr_fit, y_tr_fit,
                eval_set=[(X_val_fit, y_val_fit)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
        else:
            # [FIX-I] benar-benar tanpa eval_set
            model.fit(X_tr_fit, y_tr_fit)

    y_pred = model.predict(X_te)

    if task_type == 'classification':
        y_prob     = model.predict_proba(X_te)
        y_prob_pos = (y_prob[:, 1] if y_prob.shape[1] == 2
                      else y_prob.max(axis=1))
        metrics    = evaluate_classification(
            y_te, y_pred, y_prob_pos,
            minority_label=cfg.get('minority_label', 1))
    else:
        y_prob_pos = None
        metrics    = evaluate_regression(y_te, y_pred)

    best_iter = (model.best_iteration_
                 if hasattr(model, 'best_iteration_')
                    and model.best_iteration_ is not None
                    and model.best_iteration_ > 0
                 else 1000)

    return model, metrics, best_iter, y_pred, y_te, y_prob_pos


# ══════════════════════════════════════════
# [N] VISUALISASI (sama seperti kode lama, tidak diubah)
# ══════════════════════════════════════════
def plot_convergence(history, gen_stats, target_name, save_path):
    gens      = [s['generation']  for s in gen_stats]
    best_hist = [s['best_global'] for s in gen_stats]
    avg_hist  = [s['avg_fitness'] for s in gen_stats]
    n_feat_h  = [s['n_features']  for s in gen_stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=BG_COLOR)
    fig.suptitle(f'Konvergensi GA — {target_name} | {METHOD}',
                 fontsize=11, fontweight='bold', color='white')

    for ax in [ax1, ax2]:
        ax.set_facecolor(PANEL_COLOR)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.tick_params(colors='#64748b')

    ax1.plot(gens, best_hist, color='#4ade80', lw=2.2, label='Best Fitness')
    ax1.plot(gens, avg_hist,  color='#94a3b8', lw=1.4, linestyle='--',
             alpha=0.7, label='Avg Fitness')
    ax1.fill_between(gens, avg_hist, best_hist, alpha=0.10, color='#4ade80')
    ax1.set_ylabel('Fitness Score', color='#64748b', fontsize=10)
    ax1.set_xlim(left=1)
    ax1.legend(framealpha=0.3, facecolor='#0f172a', fontsize=9, labelcolor='#94a3b8')

    ax2.plot(gens, n_feat_h, color='#a78bfa', lw=2.0, label='N Fitur Terpilih')
    ax2.fill_between(gens, n_feat_h, alpha=0.15, color='#a78bfa')
    ax2.set_xlabel('Generasi', color='#64748b', fontsize=10)
    ax2.set_ylabel('Jumlah Fitur', color='#64748b', fontsize=10)
    ax2.set_xlim(left=1)
    ax2.legend(framealpha=0.3, facecolor='#0f172a', fontsize=9, labelcolor='#94a3b8')

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"         Plot konvergensi -> {save_path}")


def plot_feature_importance(model, selected_names, target_name,
                             save_path, top_n=30):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    n           = min(len(selected_names), top_n)
    top_idx     = indices[:n]
    top_imp     = importances[top_idx]
    top_names   = [selected_names[i] for i in top_idx]
    if top_imp.max() > 0:
        top_imp = top_imp / top_imp.max()
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, n))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.33)), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.barh(np.arange(n), top_imp[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(top_names[::-1], fontsize=8, color='#94a3b8')
    ax.set_xlabel('Relative Importance (normalized)', color='#64748b', fontsize=10)
    ax.set_title(f'Feature Importance — {target_name} | {METHOD}\n'
                 f'Top-{n} dari {len(selected_names)} fitur terpilih',
                 fontsize=11, fontweight='bold', color='white', pad=10)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.tick_params(colors='#64748b')
    ax.grid(True, axis='x', linestyle='--', alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"         Feature importance -> {save_path}")


def plot_ga_vs_baseline(ga_cv, baseline_cv, target_name,
                         save_path, task_type, wilcoxon_p=None):
    metric_lbl = ('F1_weighted (CV)' if task_type == 'classification' else 'R^2 (CV)')
    ga_mean  = np.mean(ga_cv)
    bl_mean  = np.mean(baseline_cv)
    ga_std   = np.std(ga_cv)
    bl_std   = np.std(baseline_cv)
    delta    = ga_mean - bl_mean

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    bp = ax.boxplot([baseline_cv, ga_cv], patch_artist=True, widths=0.4,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], ['#94a3b8', '#a78bfa']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'fliers']:
        for item in bp[element]:
            item.set(color='#64748b', linewidth=1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f'Baseline\n{bl_mean:.4f}+/-{bl_std:.4f}',
         f'GA Selected\n{ga_mean:.4f}+/-{ga_std:.4f}'],
        fontsize=9, color='#94a3b8')
    ax.set_ylabel(metric_lbl, color='#64748b', fontsize=10)
    ax.tick_params(colors='#64748b')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.2)
    title = f'GA vs Baseline — {target_name} | {METHOD}\nDelta = {delta:+.4f}'
    if wilcoxon_p is not None:
        sig   = ('***' if wilcoxon_p < 0.001 else '**' if wilcoxon_p < 0.01
                 else '*' if wilcoxon_p < 0.05 else 'ns')
        title += f' | Wilcoxon p={wilcoxon_p:.4f} ({sig})'
    ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"         GA vs Baseline -> {save_path}")


def plot_confusion_matrix(cm, class_names, title, save_path, metrics):
    fig, ax = plt.subplots(figsize=(5, 4.6), facecolor='white')
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f'P {c}' for c in class_names], fontsize=9, fontweight='bold')
    ax.set_yticklabels([f'T {c}' for c in class_names], fontsize=9, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    summary = (f"Acc={metrics.get('Accuracy','-')}  "
               f"F1_w={metrics.get('F1_weighted','-')}  "
               f"AUC={metrics.get('AUC','-')}")
    fig.text(0.5, 0.01, summary, ha='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eef2f7', alpha=0.95))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"         Confusion matrix -> {save_path}")


def plot_roc_curve(y_true, y_prob, title, save_path, auc_val):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='white')
    ax.set_facecolor('#f8fafc')
    ax.plot(fpr, tpr, color='#2563eb', lw=2.2,
            label=f'ROC (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=1.4, linestyle='--',
            label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2563eb')
    youden = np.argmax(tpr - fpr)
    ax.scatter(fpr[youden], tpr[youden], color='#dc2626', s=80,
               zorder=5, label='Optimal Youden')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"         ROC curve -> {save_path}")


# ══════════════════════════════════════════
# [O] LOOP UTAMA
# ══════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  MULAI — {len(TARGET_CONFIG)} TARGET x 1 METHOD")
print(f"  Method : {METHOD}")
print(f"  CR={GA_PARAMS['crossover_rate']}  MR=1/n_features (dinamis)  "
      f"Pop={GA_PARAMS['pop_size']}  Gen={GA_PARAMS['n_generations']}  "
      f"Folds={GA_PARAMS['n_folds']}  EarlyStop={GA_PARAMS['early_stop_gen']}")
print(f"{'='*65}")

all_results = []
total_start = time.time()
n_done = n_skipped = n_error = 0

for target_name, cfg in TARGET_CONFIG.items():

    target_folder = os.path.join(SPLIT_FOLDER, cfg['folder'])
    save_dir      = os.path.join(RESULT_FOLDER, target_name, METHOD)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  TARGET : {target_name.upper()} | {cfg['desc']}")
    print(f"  Tipe   : {cfg['type'].upper()}  |  "
          f"SMOTE: {'Ya' if cfg['use_smote'] else 'Tidak'}")
    print(f"{'='*65}")

    if not os.path.exists(target_folder):
        print(f"  SKIP: folder tidak ditemukan -> {target_folder}")
        continue

    done_path = os.path.join(save_dir, 'metrics_ga_fixed.csv')

    if os.path.exists(done_path):
        try:
            df_old  = pd.read_csv(done_path)
            skip_ok = True
            for key, cfg_val in [
                ('crossover_rate', GA_PARAMS['crossover_rate']),
                ('pop_size',       GA_PARAMS['pop_size']),
                ('early_stop_gen', GA_PARAMS['early_stop_gen']),
            ]:
                if key in df_old.columns:
                    old_val = float(df_old[key].iloc[0])
                    if abs(old_val - cfg_val) > 1e-9:
                        print(f"  Parameter {key} berbeda -> jalankan ulang")
                        skip_ok = False
                        break
            if skip_ok:
                print(f"  [SKIP] Sudah selesai dengan parameter sama")
                all_results.append(df_old.iloc[0].to_dict())
                n_skipped += 1
                continue
            else:
                os.remove(done_path)
        except Exception as e:
            print(f"  PERINGATAN baca hasil lama: {e}")

    t_start = time.time()

    try:
        print(f"\n         [1] Load tuned params...")
        tuned_params, is_tuned = load_tuned_params(target_name, METHOD)
        src_label = 'RSCV Tuned' if is_tuned else 'DEFAULT'
        print(f"         Sumber: {src_label}")

        lgb_params_fitness = build_lgb_params(
            tuned_params, cfg['type'],
            device=LGB_DEVICE_FITNESS, n_jobs=1)

        print(f"\n         [2] Load data split 90:10...")
        X_train, X_test, y_train, y_test = load_split_data(target_folder, METHOD)
        if X_train is None:
            n_error += 1
            continue

        feature_names = X_train.columns.tolist()

        # [FIX-A] JANGAN impute di sini. Biarkan NaN masuk raw ke fitness.
        # Hanya replace +/- inf -> NaN agar konsisten.
        X_tr_raw = X_train.replace([np.inf, -np.inf], np.nan)
        X_te_raw = X_test.replace([np.inf, -np.inf], np.nan)
        X_arr    = X_tr_raw.values.astype(np.float64)
        y_arr    = np.array(y_train).ravel()

        print(f"\n         [3] Baseline (semua fitur, imputasi per-fold)...")
        baseline_cv, baseline_mean, baseline_std = run_baseline(
            X_arr, y_arr, cfg['type'], lgb_params_fitness,
            n_folds=GA_PARAMS['n_folds'], use_smote=cfg['use_smote'])
        pd.DataFrame({
            'fold'      : list(range(1, len(baseline_cv) + 1)),
            'cv_score'  : baseline_cv,
            'n_features': len(feature_names),
        }).to_csv(os.path.join(save_dir, 'baseline_cv_scores.csv'), index=False)

        print(f"\n         [4] GA Feature Selection (PARALEL, loky)...")
        (best_chrom, selected_idx, selected_names,
         history, gen_stats, best_cv_scores) = run_ga(
            X_arr, y_arr,
            task_type     = cfg['type'],
            feature_names = feature_names,
            lgb_params    = lgb_params_fitness,
            target_name   = target_name,
            use_smote     = cfg['use_smote'],
        )

        cv_mean = float(np.mean(best_cv_scores)) if best_cv_scores else float('nan')
        cv_std  = float(np.std(best_cv_scores))  if best_cv_scores else float('nan')
        cv_str  = (f"{cv_mean:.4f} +/- {cv_std:.4f}"
                   if not np.isnan(cv_mean) else 'N/A')

        wilcoxon_p = wilcoxon_stat = None
        if (best_cv_scores and len(best_cv_scores) == len(baseline_cv)
                and len(best_cv_scores) >= 2):
            try:
                stat, p       = wilcoxon(best_cv_scores, baseline_cv)
                wilcoxon_stat = round(float(stat), 4)
                wilcoxon_p    = round(float(p),    4)
                sig = ('***' if p < 0.001 else '**' if p < 0.01
                       else '*' if p < 0.05 else 'ns')
                print(f"         Wilcoxon: stat={stat:.4f}, p={p:.4f} ({sig})")
                print(f"         LIMITASI: n=5 fold -> statistical power sangat rendah.")
            except Exception as e:
                print(f"         Wilcoxon gagal: {e}")

        reduction_pct = (1 - len(selected_idx) / len(feature_names)) * 100

        print(f"\n         [5] Model final (device={LGB_DEVICE_FINAL.upper()})...")
        print(f"         Fitur: {len(feature_names)} -> {len(selected_idx)} "
              f"(reduksi {reduction_pct:.1f}%)")

        # [FIX-B,C,I] train_final_model sudah di-fix
        (model, metrics, best_iter_final,
         y_pred, y_te, y_prob_pos) = train_final_model(
            X_tr_raw, X_te_raw, y_train, y_test,   # raw (mengandung NaN)
            selected_idx, cfg['type'], cfg,
            tuned_params, use_smote=cfg['use_smote'])

        elapsed = time.time() - t_start

        # Simpan artefak
        model.booster_.save_model(
            os.path.join(save_dir, 'model_ga_fixed.txt'))

        pd.DataFrame({'feature': feature_names, 'selected': best_chrom}).to_csv(
            os.path.join(save_dir, 'ga_chromosome.csv'), index=False)

        pd.DataFrame({'feature_idx': selected_idx, 'feature_name': selected_names}).to_csv(
            os.path.join(save_dir, 'ga_selected_features.csv'), index=False)

        pd.DataFrame(gen_stats).to_csv(
            os.path.join(save_dir, 'ga_history.csv'), index=False)

        if best_cv_scores:
            pd.DataFrame({'fold': list(range(1, len(best_cv_scores)+1)),
                          'cv_score': best_cv_scores}).to_csv(
                os.path.join(save_dir, 'ga_cv_scores_best.csv'), index=False)

        pred_df = (pd.DataFrame({'y_true': y_te, 'y_pred': y_pred, 'y_prob': y_prob_pos})
                   if cfg['type'] == 'classification' and y_prob_pos is not None
                   else pd.DataFrame({'y_true': y_te, 'y_pred': y_pred}))
        pred_df.to_csv(os.path.join(save_dir, 'predictions_ga_fixed.csv'), index=False)

        feat_imp = pd.DataFrame({
            'feature'   : selected_names,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        if feat_imp['importance'].max() > 0:
            feat_imp['importance_normalized'] = (
                feat_imp['importance'] / feat_imp['importance'].max())
        else:
            feat_imp['importance_normalized'] = 0.0
        feat_imp.to_csv(os.path.join(save_dir, 'feature_importance_ga.csv'), index=False)

        mutation_rate_used = compute_mutation_rate(len(feature_names))
        pd.DataFrame([
            {'parameter': k, 'value': v} for k, v in GA_PARAMS.items()
        ] + [
            {'parameter': 'mutation_rate_used', 'value': round(mutation_rate_used, 6)},
            {'parameter': 'n_features',         'value': len(feature_names)},
            {'parameter': 'model',              'value': 'LightGBM'},
            {'parameter': 'ga_type',            'value': 'Elitist Generational GA'},
            {'parameter': 'method',             'value': METHOD},
            {'parameter': 'seed',               'value': GLOBAL_SEED},
            {'parameter': 'params_source',      'value': src_label},
            {'parameter': 'device_fitness',     'value': 'cpu (paralel aman)'},
            {'parameter': 'device_final',       'value': LGB_DEVICE_FINAL},
            {'parameter': 'parallel_backend',   'value': 'joblib loky'},
            {'parameter': 'selection_ref',      'value': 'Roulette shift+eps (Holland 1975)'},
            {'parameter': 'crossover_ref',      'value': 'Uniform (Katoch 2021)'},
            {'parameter': 'mutation_ref',       'value': 'Bit-flip MR=1/L (Back 1997)'},
            {'parameter': 'elitism_ref',        'value': 'Katoch (2021) Sec 3.2.2'},
            {'parameter': 'nan_strategy',       'value': 'imputasi per-fold, median fold train'},
            {'parameter': 'smote_strategy',     'value': 'in-fold CV + post-split final'},
            {'parameter': 'fix_leakage_A',      'value': 'Hapus global imputation'},
            {'parameter': 'fix_leakage_B',      'value': 'Median dari X_tr_fit di final model'},
            {'parameter': 'fix_leakage_C',      'value': 'SMOTE post-split val'},
            {'parameter': 'fix_roulette_D',     'value': 'Shift+epsilon, no zero prob'},
            {'parameter': 'fix_r2_E',           'value': 'R^2 tidak di-clamp'},
            {'parameter': 'fix_mr_F',           'value': 'MR=1/n_features dinamis'},
            {'parameter': 'fix_estop_G',        'value': 'early_stop=20'},
            {'parameter': 'fix_fold_err_H',     'value': 'fold error score=-1e3'},
            {'parameter': 'fix_eval_I',         'value': 'no early-stop kalau split gagal'},
            {'parameter': 'fix_cache_J',        'value': 'fitness cache aktif'},
        ]).to_csv(os.path.join(save_dir, 'ga_params_config.csv'), index=False)

        row = {
            'target'          : target_name,
            'desc'            : cfg['desc'],
            'type'            : cfg['type'],
            'method'          : METHOD,
            'model'           : 'LightGBM',
            'ga_type'         : 'Elitist Generational GA',
            'split'           : '90_10',
            'params_source'   : src_label,
            'crossover_rate'  : GA_PARAMS['crossover_rate'],
            'mutation_rate'   : round(mutation_rate_used, 6),
            'pop_size'        : GA_PARAMS['pop_size'],
            'n_generations'   : GA_PARAMS['n_generations'],
            'early_stop_gen'  : GA_PARAMS['early_stop_gen'],
            'n_feat_original' : len(feature_names),
            'n_feat_ga'       : len(selected_idx),
            'reduction_pct'   : round(reduction_pct, 2),
            'ga_best_fitness' : round(history[-1], 5),
            'best_iter_final' : best_iter_final,
            'elapsed_min'     : round(elapsed / 60, 2),
            **metrics,
            'CV_score_mean'   : round(cv_mean, 4) if not np.isnan(cv_mean) else None,
            'CV_score_std'    : round(cv_std,  4) if not np.isnan(cv_std)  else None,
            'CV_mean_std'     : cv_str,
            'n_folds'         : GA_PARAMS['n_folds'],
            'Baseline_CV_mean': round(baseline_mean, 4),
            'Baseline_CV_std' : round(baseline_std,  4),
            'Delta_CV'        : round(cv_mean - baseline_mean, 4) if not np.isnan(cv_mean) else None,
            'Wilcoxon_stat'   : wilcoxon_stat,
            'Wilcoxon_p'      : wilcoxon_p,
        }

        pd.DataFrame([row]).to_csv(done_path, index=False)
        all_results.append(row)

        # Plot
        plot_convergence(history, gen_stats, target_name,
                         os.path.join(save_dir, 'ga_convergence.png'))
        plot_feature_importance(model, selected_names, target_name,
                                os.path.join(save_dir, 'feature_importance_ga.png'))
        if best_cv_scores:
            plot_ga_vs_baseline(best_cv_scores, baseline_cv, target_name,
                                os.path.join(save_dir, 'ga_vs_baseline.png'),
                                cfg['type'], wilcoxon_p=wilcoxon_p)

        if cfg['type'] == 'classification':
            cnames = cfg.get('class_names', ['Class 0', 'Class 1'])
            cm     = confusion_matrix(y_te, y_pred)
            plot_confusion_matrix(
                cm, cnames,
                title=f"{cfg['desc']}\n[LightGBM-GA-Fixed] ({METHOD}) 90:10",
                save_path=os.path.join(save_dir, 'confusion_matrix_ga.png'),
                metrics=metrics)
            pd.DataFrame(cm,
                index=[f'True_{c}' for c in cnames],
                columns=[f'Pred_{c}' for c in cnames]).to_csv(
                os.path.join(save_dir, 'confusion_matrix_ga.csv'))
            pd.DataFrame(
                classification_report(y_te, y_pred, target_names=cnames,
                                      output_dict=True, zero_division=0)
            ).transpose().to_csv(
                os.path.join(save_dir, 'classification_report_ga.csv'))
            if y_prob_pos is not None and not np.isnan(metrics.get('AUC', float('nan'))):
                plot_roc_curve(
                    y_te, y_prob_pos,
                    title=f"{cfg['desc']}\n[LightGBM-GA-Fixed] ({METHOD}) 90:10",
                    save_path=os.path.join(save_dir, 'roc_curve_ga.png'),
                    auc_val=metrics.get('AUC', 0.0))
                fpr_r, tpr_r, _ = roc_curve(y_te, y_prob_pos)
                pd.DataFrame({'fpr': fpr_r, 'tpr': tpr_r}).to_csv(
                    os.path.join(save_dir, 'roc_data_ga.csv'), index=False)

        delete_checkpoint(save_dir, target_name)
        n_done += 1

        print(f"\n         SELESAI: {target_name}")
        print(f"            Waktu        : {elapsed/60:.1f} menit")
        print(f"            Fitur        : {len(feature_names)} -> {len(selected_idx)} "
              f"(reduksi {reduction_pct:.1f}%)")
        print(f"            CV           : {cv_str}")
        print(f"            Baseline CV  : {baseline_mean:.4f} +/- {baseline_std:.4f}")
        if not np.isnan(cv_mean):
            print(f"            Delta CV     : {cv_mean - baseline_mean:+.4f}")
        for k, v in metrics.items():
            print(f"            {k:15s}: {v}")

    except Exception as e:
        print(f"\n         ERROR pada {target_name}: {e}")
        import traceback; traceback.print_exc()
        n_error += 1


# ══════════════════════════════════════════
# [P] RINGKASAN AKHIR
# ══════════════════════════════════════════
df_results    = pd.DataFrame(all_results) if all_results else pd.DataFrame()
total_elapsed = time.time() - total_start

print(f"\n{'='*65}")
print(f"  RINGKASAN AKHIR")
print(f"{'='*65}")
print(f"  Model           : LightGBM")
print(f"  Tipe GA         : Elitist Generational GA (build sendiri)")
print(f"  Method          : {METHOD}")
print(f"  Total target    : {len(TARGET_CONFIG)}")
print(f"  Selesai baru    : {n_done}")
print(f"  Di-skip (lama)  : {n_skipped}")
print(f"  Error           : {n_error}")
print(f"  Waktu total     : {total_elapsed/60:.1f} menit")
print(f"{'='*65}")

if not df_results.empty:
    summary_path = os.path.join(RESULT_FOLDER, 'SUMMARY_GA_FIXED.csv')
    df_results.to_csv(summary_path, index=False)
    print(f"\n  Summary -> {summary_path}")

    df_c = df_results[df_results['type'] == 'classification']
    if not df_c.empty:
        print(f"\n{'='*65}")
        print(f"  KLASIFIKASI")
        print(f"{'='*65}")
        cols = [c for c in ['target','n_feat_original','n_feat_ga',
                'reduction_pct','CV_mean_std','Accuracy',
                'F1_weighted','AUC','Baseline_CV_mean',
                'Delta_CV','Wilcoxon_p'] if c in df_c.columns]
        print(df_c[cols].to_string(index=False))

    df_r = df_results[df_results['type'] == 'regression']
    if not df_r.empty:
        print(f"\n{'='*65}")
        print(f"  REGRESI")
        print(f"{'='*65}")
        cols = [c for c in ['target','n_feat_original','n_feat_ga',
                'reduction_pct','CV_mean_std','MAE','RMSE','R2',
                'Baseline_CV_mean','Delta_CV','Wilcoxon_p']
                if c in df_r.columns]
        print(df_r[cols].to_string(index=False))

print(f"\n{'='*65}")
print(f"  GA FEATURE SELECTION — SELESAI (ALL FIXES APPLIED)")
print(f"  [FIX-A] Leakage global imputation dihapus")
print(f"  [FIX-B] Median dari X_tr_fit di final model")
print(f"  [FIX-C] SMOTE post-split val")
print(f"  [FIX-D] Roulette shift+epsilon (no zero prob)")
print(f"  [FIX-E] R^2 tidak di-clamp (nilai negatif dipertahankan)")
print(f"  [FIX-F] Mutation rate = 1/n_features (Back 1997)")
print(f"  [FIX-G] early_stop = 20 generasi")
print(f"  [FIX-H] Fold error score = -1e3 (bukan 0)")
print(f"  [FIX-I] No early-stop kalau split val gagal")
print(f"  [FIX-J] Fitness cache aktif")
print(f"{'='*65}")
