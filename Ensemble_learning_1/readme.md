# Notebook Ensemble Learning

## Gambaran Umum
Repositori ini berisi notebook Jupyter **Ensemble_learning.ipynb** yang menunjukkan cara membangun dan mengevaluasi model ensembel untuk klasifikasi biner pada dataset **Telco Customer Churn**. Notebook ini menjelaskan proses pemuatan data, pra‑pemrosesan, penanganan ketidakseimbangan kelas dengan SMOTE, pelatihan tiga classifier ensembel (Random Forest, XGBoost, LightGBM), serta evaluasi kinerjanya menggunakan metrik klasifikasi dan ROC AUC.

## Dataset
- Sumber: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7043 baris, 21 fitur (termasuk kolom target `Churn`).
- Fitur meliputi tipe kategorikal (misalnya gender, Partner) dan numerik (misalnya tenure, MonthlyCharges).

## Langkah Pra‑pemrosesan
1. **Label Encoding** – Mengubah semua kolom kategorikal menjadi label numerik (kecuali target `Churn`).
2. **Feature Scaling** – Menstandarisasi fitur numerik (`tenure`, `MonthlyCharges`, `TotalCharges`).
3. **Train‑Test Split** – Membagi data 80/20 dengan `random_state=42` tetap.
4. **Penanganan Imbalance** – Menerapkan **SMOTE** pada set pelatihan untuk menyeimbangkan kelas.

## Model yang Dilatih
| Model | Library | Parameter Utama |
|-------|---------|----------------|
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `random_state=42` |
| XGBoost | `xgboost.XGBClassifier` | default (bisa disesuaikan) |
| LightGBM | `lightgbm.LGBMClassifier` | default (bisa disesuaikan) |

Setiap model dilatih pada data tidak seimbang asli **dan** data yang telah diseimbangkan dengan SMOTE untuk memperlihatkan dampak oversampling.

## Evaluasi
Notebook ini menampilkan laporan klasifikasi lengkap (`precision`, `recall`, `f1-score`) serta skor **ROC AUC** untuk tiap model. Contoh output (Random Forest pada data asli):
```
precision    recall  f1-score   support

0       0.83      0.91      0.87      1036
1       0.66      0.48      0.55       373

accuracy                           0.80      1409
macro avg       0.74      0.69      0.71      1409
weighted avg    0.78      0.80      0.78      1409

ROC AUC: 0.6946
```
Tabel serupa ditampilkan untuk XGBoost dan LightGBM.

## Cara Menjalankan Notebook
1. **Instal dependensi** (disarankan dalam environment conda baru):
   ```bash
   pip install pandas scikit-learn imbalanced-learn xgboost lightgbm notebook
   ```
2. **Jalankan JupyterLab / Notebook** di direktori proyek:
   ```bash
   jupyter notebook
   ```
3. Buka `Ensemble_learning.ipynb` dan jalankan sel‑sel secara berurutan.

## Struktur Proyek
```
├─ Ensemble_learning.ipynb   # Notebook utama (inti repositori)
├─ WA_Fn-UseC_-Telco-Customer-Churn.csv  # File dataset
└─ README.md                 # File ini
```

