
# Laporan Proyek Machine Learning - Chalika Vanya Resya

## Domain Proyek
Dalam industri retail, efisiensi rantai pasok (*supply chain*) memiliki peran penting terhadap kelancaran operasional dan pengambilan keputusan bisnis. Salah satu aspek krusial yang mendukung proses ini adalah kemampuan bisnis/perusahaan untuk bisa memproyeksikan permintaan (*demand*) pelanggan dengan akurat. Peramalan permintaan (*demand forecasting*) yang tepat memungkinkan bisnis/perusahaan untuk mengelola inventori secara optimal, memastikan ketersediaan produk sesuai kebutuhan konsumen di waktu yang tepat, serta mencegah risiko kelebihan atau kekurangan stok. Hasil *forecasting* terhadap *demand* ini memengaruhi berbagai keputusan penting dalam rantai pasok, seperti pengadaan bahan baku, perencanaan produksi, distribusi logistik, hingga strategi pengelolaan persediaan. Prediksi yang akurat tidak hanya akan meningkatkan efisiensi, tetapi juga berdampak langsung terhadap kepuasan pelanggan dan pengurangan biaya operasional [[1]](https://www.scirp.org/journal/paperinformation?paperid=129742). 

Namun, menghasilkan proyeksi yang presisi bukanlah hal yang mudah, terutama di tengah dinamika pasar modern yang dipengaruhi oleh perkembangan teknologi, perubahan kondisi ekonomi, dan meningkatnya ekspektasi pelanggan [[2]](https://www.journal.oscm-forum.org/publication/article/applications-of-artificial-intelligence-for-demand-forecasting). Seiring meningkatnya kompleksitas ini, penerapan *Artificial Intelligence (AI)* dan pembelajaran mesin (*machine learning*) semakin banyak digunakan dalam proses *demand forecasting*. Berbagai studi menunjukkan bahwa AI memiliki kontribusi besar dalam meningkatkan kualitas prediksi permintaan dan telah menjadi pendekatan yang menjanjikan di berbagai sektor, termasuk ritel [[2]](https://www.journal.oscm-forum.org/publication/article/applications-of-artificial-intelligence-for-demand-forecasting).

Dengan demikian, proyek ini berfokus pada **pengembangan model prediksi *demand forecasting* berbasis *machine learning*** yang diharapkan mampu membantu pelaku industri retail dalam mengantisipasi kebutuhan pelanggan, meminimalkan risiko *overstock* dan *stockout*, serta mengoptimalkan rantai pasok secara keseluruhan.
 
## Business Understanding

### Problem Statements
Berdasarkan latar belakang proyek, dirumuskan beberapa permasalahan untuk diselesaikan pada proyek ini, yaitu:
- Bagaimana model dapat **memprediksi jumlah unit produk yang akan terjual** (`units_sold`), baik secara keseluruhan maupun berdasarkan toko ritel dengan akurat dengan mempertimbangkan
    - Karakteristik produk dan toko (kategori, harga, diskon, lokasi, dll),
    - Faktor eksternal seperti cuaca, musim, promosi, dan harga kompetitor,
    - Tren dan pola permintaan historis.
- Bagaimana model bisa menghasilkan kesalahan (error) terhadap data aktual yang sekecil mungkin dengan memanfaatkan metrik, seperti MSE dan RMSE?

### Goals
Berdasarkan rumusan masalah yang dipaparkan, ditentukan beberapa tujuan dari proyek ini, yaitu:
- Memberikan wawasan berbasis data untuk membantu pengambilan keputusan bisnis, khususnya dalam pengelolaan stok berdasarkan hasil *forecasting*.
- Mengembangkan model prediktif berbasis data historis penjualan dan faktor eksternal untuk memperkirakan permintaan harian produk.
- Membandingkan performa beberapa model regresi untuk memilih model terbaik yang mampu meminimalkan kesalahan prediksi.

### Solution Statements
Untuk menyelesaikan masalah *demand forecasting*, proyek ini menggunakan pendekatan *supervised machine learning* dengan memanfaatkan **tiga model regresi**, yaitu:

1. **Linear Regression**  
   - Model *baseline* yang sederhana dan mudah diinterpretasikan.
   - Digunakan sebagai pembanding awal terhadap model lain.

2. **Random Forest Regressor (with *hyperparameter tuning*)**  
   - Model *ensemble* berbasis *decision tree* yang menggunakan algoritma *bagging*.
   - Cocok menangani non-linearitas dan interaksi antar fitur.

3. **XGBoost Regressor (with *hyperparameter tuning*)**  
   - Model *gradient boosting* yang efisien terhadap data besar dan cenderung menghasilkan performa tinggi.
   - Memiliki regularisasi, seperti *shrinkage* dan pruning untuk mencegah *overfitting*.

#### Evaluation Metrics

Untuk mengevaluasi performa ketiga model, digunakan metrik berikut:

- **Mean Squared Error (MSE)**: Menghitung rata-rata dari kuadrat selisih antara prediksi dan nilai aktual.
- **Root Mean Squared Error (RMSE)**: Akar dari MSE, mengembalikan skala error ke satuan aslinya sehingga memudahkan interpretasi.

Selain evaluasi keseluruhan, dilakukan juga **analisis performa per *store*** untuk memastikan bahwa model bekerja konsisten di berbagai lokasi toko.

## Data Understanding
Data yang digunakan dalam proyek ini berasal dari dataset [Retail Store Inventory Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset/data) yang tersedia di platform Kaggle. *Dataset* ini berupa file `.csv` dengan ukuran 6.19 MB dan bernama `retail_store_inventory.csv`. Meskipun merupakan data sintetik, dataset ini dirancang agar tetap realistis dan merepresentasikan kondisi operasional retail dari tanggal 01-01-2022 hingga 01-01-2024, mencakup total 73.100 baris data dari berbagai toko dengan 15 fitur awal sebelum dilakukan *feature extraction*.

### 🧾 Deskripsi Variabel

| No | Kolom                | Tipe Data | Deskripsi |
|----|----------------------|-----------|-----------|
| 1  | `Date`               | object    | Tanggal transaksi harian |
| 2  | `Store ID`           | object    | ID unik toko |
| 3  | `Product ID`         | object    | ID unik produk |
| 4  | `Category`           | object    | Kategori produk (misalnya: Electronics, Clothing, Groceries) |
| 5  | `Region`             | object    | Wilayah geografis toko |
| 6  | `Inventory Level`    | int64     | Stok yang tersedia pada awal hari |
| 7  | `Units Sold`         | int64     | Jumlah unit terjual dalam sehari (**target variabel**) |
| 8  | `Units Ordered`      | int64     | Jumlah stok yang dipesan ulang |
| 9  | `Demand Forecast`    | float64   | Perkiraan permintaan berdasarkan tren historis |
| 10 | `Price`              | float64   | Harga jual produk |
| 11 | `Discount`           | int64     | Besaran diskon (dalam persen) |
| 12 | `Weather Condition`  | object    | Kondisi cuaca harian |
| 13 | `Holiday/Promotion`  | int64     | Indikator promosi atau hari libur (0 = Tidak, 1 = Ya) |
| 14 | `Competitor Pricing` | float64   | Harga produk di toko kompetitor |
| 15 | `Seasonality`        | object    | Musim dalam setahun (Autumn, Spring, Winter, Summer) |

---

### 📈 Statistik Deskriptif (Variabel Numerik)

| Statistik | Inventory Level | Units Sold | Units Ordered | Demand Forecast | Price | Discount | Holiday/Promotion | Competitor Pricing |
|-----------|----------------:|-----------:|---------------:|-----------------:|------:|---------:|------------------:|-------------------:|
| Count     | 73100           | 73100      | 73100          | 73100            | 73100 | 73100    | 73100             | 73100              |
| Mean      | 274.47          | 136.46     | 110.00         | 141.49           | 55.14 | 10.01     | 0.497             | 55.15              |
| Std       | 129.95          | 108.92     | 52.28          | 109.25           | 26.02 | 7.08      | 0.50              | 26.19              |
| Min       | 50              | 0          | 20             | -9.99            | 10    | 0         | 0                 | 5.03               |
| 25%       | 162             | 49         | 65             | 53.67            | 32.65 | 5         | 0                 | 32.68              |
| 50%       | 273             | 107        | 110            | 113.02           | 55.05 | 10        | 0                 | 55.01              |
| 75%       | 387             | 203        | 155            | 208.05           | 77.86 | 15        | 1                 | 77.82              |
| Max       | 500             | 499        | 200            | 518.55           | 100   | 20        | 1                 | 104.94             |

---

### 🧮 Statistik Ringkasan (Variabel Kategorikal)

| Kolom              | Jumlah Unik | Nilai Terbanyak (Top) | Frekuensi |
|--------------------|-------------|------------------------|-----------|
| `Date`             | 731         | 2024-01-01             | 100       |
| `Store ID`         | 5           | S001                   | 14620     |
| `Product ID`       | 20          | P0001                  | 3655      |
| `Category`         | 5           | Furniture              | 14699     |
| `Region`           | 4           | East                   | 18349     |
| `Weather Condition`| 4           | Sunny                  | 18290     |
| `Seasonality`      | 4           | Spring                 | 18317     |

---

### Exploratory Data Analysis (EDA)
Untuk memahami karakteristik data secara menyeluruh, dilakukan eksplorasi awal yang mencakup:

1. **Pemeriksaan Struktur Dataset**  
   Meliputi jumlah baris dan kolom, tipe data, serta identifikasi nilai null (*missing values*). Setelah diperiksa, tidak terdapat *missing values* pada *dataset* sehingga tidak perlu ada penanganan.

2. **Deteksi Outliers**  
   Visualisasi boxplot digunakan untuk melihat kemunculan nilai ekstrem pada kolom numerik.
   ![outliers](https://github.com/user-attachments/assets/ee464975-bb95-4cda-9033-dea0cb4fddd1)
   Visualisasi di atas menunjukkan bahwa *outliers* hanya ditemukan pada kolom `units_sold` dan `demand_forecast`, dengan posisi di atas *upper boundary*.

4. **Analisis Univariat**  
   Pemeriksaan distribusi setiap variabel numerik, termasuk visualisasi histogram dan tren *time-series*.
   ![time-series](https://github.com/user-attachments/assets/4cc62858-62a4-4b40-8329-5519e7ddb2d3)
   Visualisasi line plot tersebut memperlihatkan bahwa penjualan harian bersifat fluktuatif tetapi masih cukup konsisten tanpa adanya tren jangka panjang yang signifikan. Meskipun ada beberapa titik puncak (*spike*) dan lembah, pola secara umum tidak menunjukkan pergerakan naik atau turun yang signifikan.

6. **Analisis Multivariat**  
   Penggunaan heatmap korelasi dan scatter plot untuk mengidentifikasi hubungan antar fitur serta potensi multikolinearitas.
   ![pairplot](https://github.com/user-attachments/assets/04e1f0df-392b-4f28-8e62-b65a1c7ac930)
   Berdasarkan pairplot, hanya `inventory_level` dan `demand_forecast` yang menunjukkan pola hubungan linear positif dengan `units_sold`, yang berarti kedua fitur tersebut memiliki potensi kuat dalam menjelaskan variasi nilai target.

## Data Preparation

Tahapan persiapan data mencakup penyesuaian nama fitur, konversi tipe data, pembersihan *outliers*, ekstraksi fitur, pembagian data latih dan uji, serta pembangunan *preprocessing pipeline*.

### 1. Perubahan nama fitur
Hal yang dilakukan mencakup mengubah penulisan menjadi *lowercase* dan mengubah spasi menjadi *underscore* (_) untuk kemudahan pemanggilan.
```py
for col in df.columns.tolist():
  renamed_col = col.lower().replace(' ', '_')
  df = df.rename(columns={col: renamed_col})
```
### 2. Konversi tipe data
Mengubah tipe data kolom `date` dari `object` menjadi `datetime` untuk memudahkan ekstraksi data waktu yang relevan, seperti hari, bulan, dan sebagainya.

```py
df['date'] = pd.to_datetime(df['date'])
```

### 3. Missing Values
Tidak terdapat *missing values* pada *dataset* sehingga tidak perlu ada penanganan.
```py
df.isnull().sum()
```

### 4. Duplicated Data
Tidak terdapat data duplikat pada *dataset* sehingga tidak perlu ada penanganan.
```py
df.duplicated().sum()
```

### 5. Outliers

Deteksi *outliers* dilakukan menggunakan **Tukey's Rule**:
```py
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

# IQR formula
IQR = Q3 - Q1
RLB = Q1 - (1.5 * IQR)
RUB = Q3 + (1.5 * IQR)
```
*Outlier* tidak dihapus, tetapi ditangani dengan ***capping***, yaitu mengganti nilai ekstrem dengan batas bawah (`RLB`) atau batas atas (`RUB`). Hal ini dilakukan karena nilai tinggi pada `units_sold` masih relevan secara bisnis (misal karena stok besar).

```py
def cap_outliers(df, num_cols):
  for col in num_cols:
    _, RLB, RUB = outliers_info(df[col])
    # clip outliers
    df[col] = df[col].clip(RLB, RUB)
  return df
```

![outliers handled](https://github.com/user-attachments/assets/7c50b0f8-4c81-4fae-ab89-c9db87b8899f)

### 6. **Feature Extraction**

Ekstraksi fitur dilakukan untuk menambah informasi untuk membantu proses prediktif menggunakan fitur lain.

```py
# a column that represents yesterday's sales
df_cleaned['lag_1'] = df_cleaned.groupby(['store_id',  'product_id'])['units_sold'].shift(1)
# a column that represents average sales per week (7 days)
df_cleaned['rolling_mean_7'] = df_cleaned.groupby(['store_id',  'product_id'])['units_sold'].shift(1).rolling(7).mean()
# weekday and weekend columns
df_cleaned['day_of_the_week'] = pd.to_datetime(df_cleaned['date']).dt.dayofweek
df_cleaned['is_weekend'] = df_cleaned['day_of_the_week'].isin([5,6]).astype(int)  # 5 for Saturday and 6 for Sunday
# price gap column
df_cleaned['price_diff'] = df_cleaned['competitor_pricing'] - df_cleaned['price']
```
Fitur-fitur baru yang dihasilkan antara lain:
- `lag_1`: penjualan 1 hari sebelumnya (per produk per toko)
- `rolling_mean_7`: rata-rata penjualan 7 hari sebelumnya
- `day_of_the_week`: indeks hari dalam seminggu
- `is_weekend`: indikator apakah hari tersebut termasuk akhir pekan
- `price_diff`: selisih harga dengan kompetitor

### 7. **Data Splitting**
Karena data bersifat *time-series*, pemisahan dilakukan berdasarkan **urutan waktu** (bukan acak), dengan rasio **90:10**:
```py
df_cleaned = df_cleaned.sort_values(by='date').reset_index(drop=True)
split_index = int(len(df_cleaned) * 0.9)
train_df = df_cleaned.iloc[:split_index]
test_df = df_cleaned.iloc[split_index:]
```
Hasil pemisahan menunjukkan titik pemisah berada pada tanggal `2023-10-20`, menghasilkan **65.790 data latih** dan **7.310 data uji**.

Dilakukan juga pemisahan fitur dengan target variabel untuk data latih dan data uji.

```py
# separate target variable from features
target_col = 'units_sold'

# 1. train data
X_train = train_df.drop(columns=target_col)
y_train = train_df[target_col]

# 2. test data
X_test = test_df.drop(columns=target_col)
y_test = test_df[target_col]
```

### 8. **Preprocessing Pipeline**
*Pipeline* digunakan untuk menyiapkan data sebelum *modeling*, meliputi:
-   **Imputasi** nilai null (khususnya fitur `lag_1` dan `rolling_mean_7`)
-   **Standarisasi** fitur numerik
-   **Encoding** fitur kategorikal; Label Encoding untuk kolom id (`store_id` & `product_id`) dan One-Hot Encoding untuk kolom kategorikal lainnya.

```py
# label encoding for ids
le_dict = dict()

for col in id_cols:
  encoder = LabelEncoder()
  X_train[col] = encoder.fit_transform(X_train[col]) + 1 # start label from 1
  le_dict[col] = encoder # save encoder for test set
```

```py
# define variables to store columns for different pipeline steps

# 1. id columns for label encoding
id_cols = ['store_id', 'product_id']

# 2. the remaining categorical columns for one-hot encoding
cat_cols = X_train.select_dtypes('object').columns.difference(id_cols).tolist()

# 3. numerical columns for scaling and imputing
num_cols = X_train.select_dtypes('number').columns.difference(id_cols).tolist()

# define pipeline transformers

# numeric pipeline
num_pipeline = Pipeline(steps=[
('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # impute nan in lag and rolling columns with 0 indicating no history
('scaler', StandardScaler())  # scaling using Standardization technique
])

# categorical pipeline
cat_pipeline = Pipeline(steps=[
('imputer', SimpleImputer(strategy='most_frequent')),
('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# combined pipelines
preprocessor = ColumnTransformer(transformers=[
('num', num_pipeline, num_cols),
('cat', cat_pipeline, cat_cols)
], remainder='passthrough')

# final pipeline
full_pipeline = Pipeline(steps=[
('preprocessor', preprocessor),
])
```

*Pipeline* ini diterapkan **terpisah untuk data latih dan uji** untuk menghindari *data leakage* dengan penggunaan syntax `.fit_transform()` hanya pada data latih.

```py
# apply pipeline on X_train
X_train_prepared = full_pipeline.fit_transform(X_train)
```
```py
# apply transformations to X_test

# 1. label encode store_id and product_id using the saved encoder
for col in id_cols:
  encoder = le_dict[col]
  X_test[col] = encoder.transform(X_test[col]) + 1

# 2. apply full pipeline to X_test
X_test_prepared = full_pipeline.transform(X_test)
```

## Modeling
Setelah data siap digunakan, dilakukan pelatihan model menggunakan tiga algoritma dengan karakteristik berbeda: **Linear Regression**, **Random Forest**, dan **XGBoost**. Model-model ini dipilih karena mewakili algoritma linear sederhana serta *ensemble learning* berbasis *bagging* dan *boosting*. 
- **Linear Regression**: model linier yang digunakan sebagai *baseline* karena sederhana dan mudah diinterpretasi. Namun, rentan *underfitting* jika data tidak memiliki pola linier[[3]](https://www.geeksforgeeks.org/ml-linear-regression/)[[4]](https://www.geeksforgeeks.org/ml-advantages-and-disadvantages-of-linear-regression/).
- **Random Forest**: model *ensemble* berbasis *bagging* dengan banyak pohon keputusan (*decision tree*). Kelebihannya adalah *robust* terhadap *outliers* dan *missing values*, serta menyediakan *feature importance*, tetapi cenderung *overfit* dan lebih berat secara komputasi [[5]](https://aiml.com/what-are-the-advantages-and-disadvantages-of-random-forest/).
- **XGBoost**:  model *ensemble* berbasis *boosting* yang dilatih secara bertahap. Model ini unggul dalam menangani data besar, cukup efisien, dan memiliki mekanisme regularisasi bawaan untuk menghindari potensi *overfit*, tetapi komputasinya intensif, interpretasi hasil yang sulit, dan rentan *overfitting* pada *dataset* kecil [[6]](https://www.geeksforgeeks.org/xgboost/).

### Pelatihan Model
Model dilatih menggunakan parameter default, kecuali `n_estimators` pada Random Forest dan XGBoost untuk membatasi waktu pelatihan:

```py
# model training: linear regression, random forest, xgboost
lr = LinearRegression().fit(X_train_prepared, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_prepared, y_train)
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42).fit(X_train_prepared, y_train)
```

Untuk evaluasi, dibuat dataframe `model_evaluation` yang menyimpan nilai evaluasi ***Mean-Squared Error (MSE)*** dan ***Root Mean-Squared Error (RMSE)*** pada data latih dan uji:
```py
model_evaluation = pd.DataFrame(columns=['train_mse', 'test_mse', 'train_rmse', 'test_rmse'],
                                index=['Linear Regression', 'Random Forest', 'XGBoost'])
```

### Hasil Evaluasi Awal
- **Linear regression**: Error tertinggi, mengindikasikan *underfitting*.
- **Random forest**: Selisih yang besar antara *train* dan *test* error. Hal ini mengindikasikan *overfitting* dan perlu dilakukan *hyperparameter tuning*.
- **XGBoost**: Selisih *train-test* lebih kecil dibanding random forest. Memiliki performa awal baik, tetapi tetap dilakukan *tuning* untuk meningkatkan performa.

### Hyperparameter tuning
Dilakukan pada model random forest dan xgboost menggunakan `RandomizedSearchCV` untuk mengurangi biaya komputasi.
1. **Random Forest**
    ```py
    rf_params = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 5]
    }
    ```
    Parameter yang ditentukan untuk *tuning*:
    - `n_estimators`: jumlah pohon keputusan dalam hutan.
    - `min_samples_split`: jumlah minimum sampel yang dibutuhkan untuk memisahkan suatu node.
    - `min_samples_leaf`: jumlah minimum sampel yang harus ada di setiap daun (*leaf node*).
    - `max_depth`: kedalaman maksimum dari masing-masing pohon keputusan.

2. **XGBoost**
    ```py
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    ```
    Parameter yang ditentukan untuk *tuning*:
    - `n_estimators`: jumlah pohon yang digunakan dalam *boosting*.
    - `max_depth`: kedalaman maksimum tiap pohon untuk mengontrol kompleksitas.
    - `learning_rate`: ukuran langkah pembelajaran
    - `subsample`: proporsi sampel data per iterasi.
    - `colsample_bytree`: proporsi fitur setiap pohon.

Model terbaik akan dipilih berdasarkan hasil evaluasi setelah *tuning* menggunakan metrik dengan error terkecil.

## Evaluation

Evaluasi model dilakukan menggunakan metrik regresi, yaitu **Mean Squared Error (MSE)** dan **Root Mean Squared Error (RMSE)**:

- **Mean Squared Error (MSE)**  
  Mengukur rata-rata selisih kuadrat antara nilai aktual dan prediksi.  
```math
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```

- **Root Mean Squared Error (RMSE)**  
  Akar kuadrat dari MSE, menyatakan deviasi prediksi dalam satuan yang sama dengan target.  
```math
\text{RMSE} = \sqrt{\text{MSE}}
```

MSE digunakan karena kemudahannya dalam melakukan operasi matematika dibandingkan MAE [[7]](https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e), sedangkan RMSE dipilih karena memiliki satuan yang sama dengan variabel target (`units_sold`) sehingga lebih mudah diinterpretasikan.

### Hasil Evaluasi Awal

| Model              | Train MSE | Test MSE | Train RMSE | Test RMSE |
|--------------------|-----------|----------|------------|-----------|
| Linear Regression  | 73.493    | 74.620   | 8.573      | 8.638     |
| Random Forest | 10.037    | 72.845   | 3.168       | 8.535     |
| XGBoost       | 55.576    | 73.891   | 7.455      | 8.596    |

Berdasarkan hasil evaluasi awal (RMSE) dan data statistik `units_sold`:
- **Rata-rata penjualan harian**: ~136 unit  
- **RMSE (~9 unit)** setara dengan error ~6.6% dari rata-rata penjualan (tergolong relatif kecil)
- **Simpangan baku penjualan harian**: ~108 unit → model cukup baik dalam menangkap fluktuasi alami

#### Ringkasan Performa Model

- **Linear Regression** → Error tertinggi, indikasi *underfitting*
- **Random Forest** → *Train* error kecil, tetapi awalnya *overfit* sebelum *tuning*
- **XGBoost** → Lebih seimbang antara *train* dan *test* error, performa stabil

### Setelah Hyperparameter Tuning

| Model              | Train MSE | Test MSE | Train RMSE | Test RMSE |
|--------------------|-----------|----------|------------|-----------|
| Linear Regression  | 73.493    | 74.620   | 8.573      | 8.638     |
| Best Random Forest | 60.517    | 70.571   | 7.779      | 8.401     |
| Best XGBoost       | 61.610    | 71.792   | 7.849      | 8.473     |

- Performa **Random Forest** dan **XGBoost** meningkat setelah *tuning*
- Selisih error *train-test* kedua model mengecil → lebih stabil
- **Best Random Forest** memberikan error terkecil dan dipilih sebagai model terbaik

### Contoh Hasil Prediksi

| y_true | pred_LR | err_LR | pred_RF | err_RF | pred_XGB | err_XGB |
|--------|---------|--------|---------|--------|----------|---------|
| 43     | 47.1    | 4.1    | 44.7    | 1.7    | 46.2     | 3.2     |
| 2      | -10.7   | 12.7   | 1.9     | 0.1    | 2.6      | 0.6     |
| 159    | 145.8   | 13.2   | 146.8   | 12.2   | 145.1    | 13.9    |
| 110    | 102.7   | 7.3    | 102.7   | 7.3    | 103.4    | 6.6     |
| 6      | -1.9    | 7.9    | 6.7     | 0.7    | 7.0      | 1.0     |

Hasil prediksi menunjukkan bahwa **Best Random Forest** secara konsisten memberikan prediksi paling akurat dan stabil.

### Evaluasi Per Store
Untuk memahami performa model secara lebih mendalam, dilakukan evaluasi berdasarkan masing-masing `store_id`. Pendekatan ini membantu mengidentifikasi apakah model bekerja secara konsisten di seluruh toko atau terdapat toko-toko tertentu dengan pola penjualan yang lebih sulit diprediksi.

Evaluasi dilakukan dengan menghitung nilai **RMSE** pada data uji untuk setiap toko.

#### Hasil RMSE per Store 
| Store ID | MSE                    | RMSE                     | MAE               |
|----------|------------------------|--------------------------|-------------------|
| 5.0      | 69.909183              | 8.361171                 | 7.068888          |
| 4.0      | 70.013739              | 8.367421                 | 7.159817          |
| 2.0      | 70.644238              | 8.405013                 | 7.168164          |
| 3.0      | 71.061937              | 8.429824                 | 7.198550          |
| 1.0      | 71.224515              | 8.439462                 | 7.216540          |

- Model terbaik (**Random Forest**) memberikan hasil prediksi yang relatif konsisten di semua store, dengan nilai RMSE yang saling berdekatan (sekitar 8.36-8.44).
- Store 5 memiliki akurasi terbaik (nilai error terkecil pada semua metrik), sedangkan store 1 memiliki akurasi terendah meskipun perbedaannya tidak signifikan.

Dengan hasil ini, **Best Random Forest tetap dipilih sebagai model utama** untuk proses prediksi karena konsistensinya dalam menghasilkan error yang rendah di sebagian besar toko.

## Final Conclusion

Proyek ini bertujuan untuk mengembangkan model prediktif yang memperkirakan jumlah unit produk yang akan terjual berdasarkan data historis, karakteristik produk dan toko, serta faktor eksternal seperti harga kompetitor dan hari dalam seminggu. Ketiga model regresi yang diujikan, yakni **Linear Regression**, **Random Forest**, dan **XGBoost**, telah dibandingkan menggunakan metrik **MSE** dan **RMSE**. Hasil evaluasi menunjukkan Random Forest setelah proses *tuning* menghasilkan performa terbaik (Test RMSE = 8.401). Evaluasi performa per toko menggunakan model ini juga menunjukkan hasil yang konsisten di seluruh toko.

Secara keseluruhan, hasil prediksi yang tergolong baik dan stabil menunjukkan bahwa model dapat digunakan untuk mendukung pengambilan keputusan bisnis, seperti pengelolaan stok dan strategi promosi. Dengan demikian, proyek ini tidak hanya menjawab *problem statement* yang diajukan, tetapi juga berhasil memenuhi seluruh tujuan dan solusi yang direncanakan secara nyata dan berdampak. Walaupun begitu, proyek ini masih dapat ditingkatkan, misalnya dengan menggunakan pendekatan *tuning* berbeda, seperti `GridSearchCV` atau menggunakan pilihan model regresi maupun *deep learning* lainnya.


## References
[1] H. Badr and W. Ahmed, "A comprehensive analysis of demand prediction models in supply chain management," SCIRP, https://www.scirp.org/journal/paperinformation?paperid=129742 (accessed May 3, 2025).  

[2] T. Nguyen, "Applications of artificial intelligence for demand forecasting," OSCM Journal, https://www.journal.oscm-forum.org/publication/article/applications-of-artificial-intelligence-for-demand-forecasting (accessed May 3, 2025). 

[3] "Linear regression in machine learning, GeeksforGeeks," https://www.geeksforgeeks.org/ml-linear-regression/ (accessed May 3, 2025). 

[4] "ML – Advantages and Disadvantages of Linear Regression," GeeksforGeeks, https://www.geeksforgeeks.org/ml-advantages-and-disadvantages-of-linear-regression/ (accessed May 3, 2025).

[5] "What are the advantages and disadvantages of Random Forest?," AIML.com, https://aiml.com/what-are-the-advantages-and-disadvantages-of-random-forest/ (accessed May 3, 2025). 

[6] "XGBoost," GeeksforGeeks, https://www.geeksforgeeks.org/xgboost/ (accessed May 3, 2025). 

[7] A. Chugh, "Mae, MSE, RMSE, coefficient of determination, adjusted R squared - which metric is better?," Medium, https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e (accessed May 3, 2025). 
