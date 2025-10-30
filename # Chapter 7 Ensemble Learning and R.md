# Chapter 7: Ensemble Learning and Random Forests

---

## 📖 Rangkuman Chapter 7

[cite_start]Chapter ini membahas **Ensemble Learning**, sebuah teknik yang menggabungkan prediksi dari sekelompok prediktor (disebut *ensemble*) untuk mendapatkan prediksi yang lebih baik daripada prediktor individu terbaik[cite: 5].

[cite_start]Ide dasarnya mirip dengan **"wisdom of the crowd"**, di mana jawaban gabungan dari ribuan orang seringkali lebih baik daripada jawaban seorang ahli[cite: 3, 4]. [cite_start]Salah satu contoh paling terkenal adalah **Random Forest**, yang merupakan ensemble dari Decision Trees[cite: 9]. [cite_start]Ensemble Learning sering digunakan untuk memenangkan kompetisi Machine Learning[cite: 11]. [cite_start]Chapter ini akan mencakup metode ensemble populer: **bagging**, **boosting**, dan **stacking**[cite: 12].

---

## 🎯 Topics Covered

| No | Topic | Description |
|----|-------|-------------|
| 1 | **Voting Classifiers** | Menggabungkan prediksi dengan suara mayoritas (hard & soft voting) |
| 2 | **Bagging & Pasting** | Melatih prediktor pada subset acak data (dengan/tanpa replacement) |
| 3 | **Out-of-Bag (oob) Evaluation** | Mengevaluasi model bagging tanpa perlu validation set |
| 4 | **Random Forests** | Ensemble Decision Trees yang dilatih dengan metode bagging |
| 5 | **Extra-Trees** | Varian Random Forest yang lebih "acak" |
| 6 | **Feature Importance** | Cara Random Forest mengukur pentingnya setiap fitur |
| 7 | **Boosting (AdaBoost)** | Melatih prediktor secara berurutan, fokus pada kesalahan sebelumnya |
| 8 | **Gradient Boosting** | Melatih prediktor secara berurutan, fokus pada *residual errors* |
| 9 | **Stacking** | Menggunakan model (blender) untuk menggabungkan prediksi ensemble |

---

## 🗳️ Voting Classifiers

[cite_start]Ini adalah cara sederhana untuk membuat classifier yang lebih baik: latih beberapa classifier yang berbeda (misalnya, Logistic Regression, SVM, Random Forest)[cite: 16], lalu gabungkan prediksi mereka.

* [cite_start]**Hard Voting Classifier**: Memprediksi kelas yang mendapatkan suara mayoritas (paling banyak dipilih) dari semua classifier[cite: 25, 26]. [cite_start]Anehnya, metode ini seringkali mencapai akurasi lebih tinggi daripada classifier terbaik di dalam ensemble[cite: 37].
* [cite_start]**Soft Voting Classifier**: Jika semua classifier dapat menghitung probabilitas (`predict_proba()`), Anda dapat memprediksi kelas dengan probabilitas rata-rata tertinggi[cite: 94]. [cite_start]Ini seringkali berkinerja lebih baik karena memberi bobot lebih pada "suara" yang sangat percaya diri[cite: 95].

**Mengapa ini berhasil? (Law of Large Numbers)**
[cite_start]Anggap Anda memiliki 1.000 classifier "lemah" (weak learners) yang hanya 51% akurat (sedikit lebih baik dari tebakan acak)[cite: 65]. [cite_start]Jika Anda mengambil suara mayoritas, akurasi ensemble bisa mencapai 75%[cite: 66]!

[cite_start]Ini berhasil dengan asumsi bahwa semua classifier **independen** dan membuat **kesalahan yang tidak berkorelasi**[cite: 67]. [cite_start]Karena itu, metode ensemble bekerja paling baik ketika prediktornya **beragam (diverse)**[cite: 69].

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

# Hard Voting
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    [cite_start]voting='hard' [cite: 80, 81]
)
[cite_start]voting_clf.fit(X_train, y_train) [cite: 82]
```

---

## 🛍️ Bagging and Pasting

Ini adalah pendekatan lain untuk mendapatkan classifier yang beragam. [cite_start]Alih-alih menggunakan algoritme yang berbeda, Anda menggunakan **algoritme yang sama** (misalnya, Decision Tree) tetapi melatihnya pada **subset acak yang berbeda** dari training set[cite: 101].

* [cite_start]**Bagging (Bootstrap Aggregating)**: Sampling dilakukan **dengan replacement** (`bootstrap=True`)[cite: 104, 131]. [cite_start]Ini berarti satu instance bisa diambil beberapa kali untuk satu prediktor[cite: 106].
* [cite_start]**Pasting**: Sampling dilakukan **tanpa replacement** (`bootstrap=False`)[cite: 105, 131].

**Cara Kerja:**
[cite_start]Setelah semua prediktor dilatih (bisa secara paralel) [cite: 127][cite_start], ensemble membuat prediksi dengan mengambil **mode** (untuk klasifikasi) atau **rata-rata** (untuk regresi) dari semua prediksi individu[cite: 119].

[cite_start]**Hasil:** Ensemble memiliki **bias yang serupa** tetapi **variance yang lebih rendah** daripada satu prediktor yang dilatih pada semua data[cite: 121].

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Ensemble Bagging dengan 500 Decision Trees
# Setiap pohon dilatih pada 100 instance acak (dengan replacement)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    [cite_start]max_samples=100, bootstrap=True, n_jobs=-1 [cite: 134, 135]
)
[cite_start]bag_clf.fit(X_train, y_train) [cite: 135]
```

### Out-of-Bag (oob) Evaluation

[cite_start]Saat menggunakan bagging (sampling dengan replacement), rata-rata hanya **63%** instance pelatihan yang di-sampling untuk setiap prediktor[cite: 161].

[cite_start]**37%** sisanya disebut **out-of-bag (oob) instances**[cite: 162]. [cite_start]Karena prediktor tidak pernah "melihat" instance oob ini selama pelatihan, kita dapat menggunakannya sebagai validation set gratis tanpa perlu memisahkan data[cite: 164].

Anda bisa mendapatkan skor oob otomatis setelah pelatihan dengan mengatur `oob_score=True`.

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    [cite_start]bootstrap=True, n_jobs=-1, oob_score=True [cite: 170-172]
)
[cite_start]bag_clf.fit(X_train, y_train) [cite: 173]

# Skor oob (evaluasi) otomatis tersedia
[cite_start]print(bag_clf.oob_score_) [cite: 174]
```
[cite_start]Skor oob biasanya sangat dekat dengan akurasi di test set[cite: 176, 180, 181].

---

## 🌲 Random Forests

[cite_start]**Random Forest** adalah ensemble dari Decision Trees, yang umumnya dilatih melalui metode **bagging** (`bootstrap=True`), dan `max_samples` biasanya diatur ke ukuran seluruh training set[cite: 206].

[cite_start]Anda dapat menggunakan kelas `RandomForestClassifier` yang lebih mudah dan teroptimasi[cite: 207].

```python
from sklearn.ensemble import RandomForestClassifier

# Melatih Random Forest dengan 500 pohon
# Setiap pohon dibatasi hingga 16 leaf nodes
[cite_start]rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1) [cite: 210, 211]
[cite_start]rnd_clf.fit(X_train, y_train) [cite: 211]
```

**Apa yang membuatnya "Random"?**
Random Forest menambahkan dua elemen keacakan:
1.  **Bagging**: Setiap pohon dilatih pada subset data acak (seperti Bagging).
2.  [cite_start]**Splitting Acak**: Saat membelah (splitting) sebuah node, alih-alih mencari fitur *terbaik* dari *semua* fitur, pohon ini mencari fitur terbaik **di antara subset fitur yang dipilih secara acak**[cite: 215].

[cite_start]Ini menghasilkan pohon yang lebih beragam, yang menukar bias yang sedikit lebih tinggi dengan variance yang jauh lebih rendah, menghasilkan model keseluruhan yang lebih baik[cite: 216].

### Extra-Trees (Extremely Randomized Trees)

Ini adalah varian yang lebih "acak" lagi. Perbedaannya:
* [cite_start]**Threshold Acak**: Alih-alih mencari *threshold* (ambang batas) terbaik untuk membelah fitur (seperti yang dilakukan Decision Tree biasa), Extra-Trees menggunakan **threshold acak** untuk setiap fitur[cite: 227].
* [cite_start]**Lebih Cepat**: Karena tidak perlu mencari threshold optimal, Extra-Trees **jauh lebih cepat** untuk dilatih daripada Random Forest biasa[cite: 230].

[cite_start]Hanya ada satu cara untuk mengetahui mana yang lebih baik (Random Forest atau Extra-Trees): coba keduanya dan lakukan cross-validation[cite: 234].

### Feature Importance

[cite_start]Kelebihan hebat dari Random Forest adalah kemudahannya mengukur **pentingnya setiap fitur**[cite: 236]. [cite_start]Scikit-Learn mengukurnya dengan melihat seberapa banyak node pohon yang menggunakan fitur tersebut berhasil **mengurangi impurity** (rata-rata di semua pohon)[cite: 237].

Skor ini dihitung secara otomatis setelah pelatihan dan dapat diakses melalui `feature_importances_`.

```python
# Latih pada data Iris
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
[cite_start]rnd_clf.fit(iris["data"], iris["target"]) [cite: 247, 248]

# Cetak skor pentingnya setiap fitur
[cite_start]for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_): [cite: 249]
    [cite_start]print(name, score) [cite: 250]
```
**Output yang Diharapkan:**
```
[cite_start]sepal length (cm) 0.112... [cite: 251]
[cite_start]sepal width (cm) 0.023... [cite: 252]
[cite_start]petal length (cm) 0.441... [cite: 253]
[cite_start]petal width (cm) 0.423... [cite: 254]
```
*(Ini menunjukkan bahwa fitur 'petal' jauh lebih penting daripada fitur 'sepal' untuk dataset Iris).*

---

## 🚀 Boosting

[cite_start]**Boosting** adalah metode ensemble yang melatih prediktor secara **berurutan**, di mana setiap prediktor baru mencoba **mengoreksi kesalahan pendahulunya**[cite: 261, 262].

[cite_start]Kelemahan utama: Boosting **tidak dapat diparalelkan** (atau hanya sebagian), karena setiap prediktor bergantung pada hasil prediktor sebelumnya[cite: 298]. [cite_start]Akibatnya, metode ini tidak dapat diskalakan sebaik bagging atau pasting[cite: 299].

### AdaBoost (Adaptive Boosting)

[cite_start]Ini adalah metode boosting yang paling populer[cite: 265].
**Cara Kerja:**
1.  [cite_start]Algoritme melatih classifier dasar (misalnya, Decision Stump: Decision Tree dengan `max_depth=1`)[cite: 270, 335].
2.  [cite_start]Ia **meningkatkan bobot (weight)** dari instance pelatihan yang **salah diklasifikasikan**[cite: 271].
3.  [cite_start]Ia melatih classifier kedua menggunakan **bobot yang telah diperbarui** (sehingga classifier kedua lebih fokus pada "kasus sulit")[cite: 272].
4.  Proses ini diulang.
5.  [cite_start]Prediksi akhir adalah **weighted vote**, di mana classifier yang lebih akurat mendapatkan bobot (weight) yang lebih tinggi[cite: 297, 313].

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost dengan 200 Decision Stumps (default base estimator)
# [cite_start]SAMME.R adalah varian yang menggunakan probabilitas, biasanya lebih baik [cite: 333]
ada_clf = AdaBoostClassifier(
    [cite_start]DecisionTreeClassifier(max_depth=1), n_estimators=200, [cite: 340]
    [cite_start]algorithm="SAMME.R", learning_rate=0.5 [cite: 340]
)
[cite_start]ada_clf.fit(X_train, y_train) [cite: 341]
```

### Gradient Boosting

[cite_start]Metode boosting populer lainnya[cite: 344].
**Cara Kerja:**
[cite_start]Alih-alih memperbarui *bobot instance* seperti AdaBoost, Gradient Boosting melatih prediktor baru untuk memperbaiki **residual errors** (selisih antara prediksi dan nilai aktual) dari prediktor sebelumnya[cite: 345].

**Contoh Regresi (GBRT):**
1.  [cite_start]Latih `tree_reg1` pada data (X, y)[cite: 355].
2.  [cite_start]Hitung residual errors: `y2 = y - tree_reg1.predict(X)`[cite: 357].
3.  [cite_start]Latih `tree_reg2` pada data (X, y2)[cite: 359].
4.  [cite_start]Hitung residual errors berikutnya: `y3 = y2 - tree_reg2.predict(X)`[cite: 361].
5.  [cite_start]Latih `tree_reg3` pada data (X, y3)[cite: 365].
6.  [cite_start]Prediksi akhir untuk instance baru adalah **jumlah prediksi** dari semua pohon: `tree_reg1.predict(X_new) + tree_reg2.predict(X_new) + tree_reg3.predict(X_new)`[cite: 367].


**Di Scikit-Learn:**
```python
from sklearn.ensemble import GradientBoostingRegressor

[cite_start]gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0) [cite: 377, 378]
[cite_start]gbrt.fit(X, y) [cite: 379]
```

**Hyperparameter Penting:**
* [cite_start]`learning_rate`: Mengatur kontribusi setiap pohon[cite: 457].
    * [cite_start]**Nilai rendah (misal, 0.1)**: Perlu lebih banyak pohon (`n_estimators`) untuk fit, tetapi biasanya **generalisasinya lebih baik**[cite: 457]. [cite_start]Ini adalah teknik regularisasi yang disebut **shrinkage**[cite: 458].
* `n_estimators`: Jumlah pohon.

**Menemukan Jumlah Pohon Optimal (Early Stopping):**
[cite_start]Anda tidak ingin terlalu sedikit pohon (underfit) atau terlalu banyak pohon (overfit)[cite: 458]. [cite_start]Anda dapat menggunakan **early stopping** untuk menemukan jumlah pohon yang optimal[cite: 481].

```python
import numpy as np
from sklearn.metrics import mean_squared_error

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
[cite_start]gbrt.fit(X_train, y_train) [cite: 489]

# Mengukur error di setiap tahap (setiap penambahan pohon)
errors = [mean_squared_error(y_val, y_pred)
          [cite_start]for y_pred in gbrt.staged_predict(X_val)] [cite: 490, 491]

# Menemukan jumlah pohon (tahap) dengan error validasi terendah
[cite_start]bst_n_estimators = np.argmin(errors) + 1 [cite: 492]

# Melatih model final dengan jumlah pohon optimal
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
[cite_start]gbrt_best.fit(X_train, y_train) [cite: 493]
```

**XGBoost (Extreme Gradient Boosting):**
[cite_start]Implementasi Gradient Boosting yang sangat populer, cepat, dan teroptimasi, yang sering memenangkan kompetisi ML[cite: 545, 547].

```python
import xgboost

[cite_start]xgb_reg = xgboost.XGBRegressor() [cite: 550, 551]
xgb_reg.fit(X_train, y_train,
            [cite_start]eval_set=[(X_val, y_val)], early_stopping_rounds=2) [cite: 555, 556]
[cite_start]y_pred = xgb_reg.predict(X_val) [cite: 557]
```

---

## 🏗️ Stacking (Stacked Generalization)

Ini adalah metode ensemble terakhir yang dibahas.

**Ide Dasar:**
[cite_start]Daripada menggunakan fungsi sederhana (seperti voting atau rata-rata) untuk menggabungkan prediksi, **mengapa kita tidak melatih model untuk melakukan agregasi tersebut?** [cite: 561]


**Cara Kerja (Menggunakan Hold-out Set):**
1.  [cite_start]**Split Data**: Bagi training set menjadi dua subset (Subset 1 dan Subset 2)[cite: 577].
2.  [cite_start]**Latih Layer 1**: Latih prediktor-prediktor (misalnya, `model_A`, `model_B`, `model_C`) hanya pada **Subset 1**[cite: 577].
3.  [cite_start]**Buat Dataset Baru (Blending)**: Gunakan prediktor-prediktor dari Layer 1 untuk membuat prediksi pada **Subset 2**[cite: 584]. [cite_start]"Prediksi bersih" ini [cite: 585] (misalnya, `pred_A`, `pred_B`, `pred_C`) menjadi fitur input baru. [cite_start]Targetnya tetap target asli dari Subset 2[cite: 591].
4.  [cite_start]**Latih Blender**: Latih model final (disebut **blender** atau **meta learner**) [cite: 563] [cite_start]pada dataset baru yang baru saja dibuat (fitur `[pred_A, pred_B, pred_C]`, target `y_subset2`)[cite: 592].

Model blender ini belajar cara terbaik untuk menggabungkan output dari prediktor Layer 1.

[cite_start]Scikit-Learn tidak mendukung stacking secara langsung [cite: 616][cite_start], tetapi Anda dapat mengimplementasikannya secara manual atau menggunakan library seperti DESlib[cite: 617].

---

## 🔧 Exercises (from the book)

### Exercise 1
**Q:** If you have trained five different models on the exact same training data, and they all achieve 95% precision, is there any chance that you can combine these models to get better results?
**A:** Ya, ada peluang. Jika kelima model **membuat tipe kesalahan yang berbeda**, menggabungkan mereka dengan *voting classifier* (terutama soft voting) dapat meningkatkan akurasi/presisi. [cite_start]Jika mereka semua membuat kesalahan yang *sama persis*, menggabungkannya tidak akan membantu[cite: 67, 71].

---

### Exercise 2
**Q:** What is the difference between hard and soft voting classifiers?
**A:**
* [cite_start]**Hard voting**: Menghitung suara dari setiap classifier dan memilih kelas yang paling banyak dipilih (mode statistik)[cite: 25, 26, 119].
* [cite_start]**Soft voting**: Menghitung rata-rata probabilitas kelas yang diprediksi dari semua classifier, lalu memilih kelas dengan probabilitas rata-rata tertinggi[cite: 94, 95]. [cite_start]Soft voting seringkali lebih baik karena memberi bobot lebih pada suara yang sangat "percaya diri"[cite: 95].

---

### Exercise 3
**Q:** Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers? What about pasting, boosting, Random Forests, or stacking?
**A:**
* **Bagging:** **Ya**. [cite_start]Setiap prediktor dilatih secara independen pada subset data yang berbeda, sehingga dapat diparalelkan dengan sempurna[cite: 127].
* **Pasting:** **Ya**. [cite_start]Sama seperti bagging, pelatihan bersifat independen dan paralel[cite: 127, 128].
* **Boosting:** **Tidak**. [cite_start]Boosting bersifat *sekuensial*; setiap prediktor baru bergantung pada hasil prediktor sebelumnya, sehingga tidak dapat diparalelkan[cite: 298, 299].
* **Random Forests:** **Ya**. [cite_start]Sama seperti bagging, setiap pohon dilatih secara independen[cite: 207].
* **Stacking:** **Sebagian**. [cite_start]Prediktor-prediktor dalam *satu layer* dapat dilatih secara paralel[cite: 577]. [cite_start]Namun, *antar layer* bersifat sekuensial (Layer 2 harus menunggu Layer 1 selesai, Blender harus menunggu Layer 1 selesai membuat prediksi)[cite: 584, 592].

---

### Exercise 4
**Q:** What is the benefit of out-of-bag evaluation?
[cite_start]**A:** Manfaat utamanya adalah Anda mendapatkan evaluasi model (mirip dengan cross-validation) secara **gratis** tanpa perlu membuat *validation set* terpisah[cite: 164]. [cite_start]Ini karena setiap prediktor dievaluasi pada instance (oob) yang tidak pernah dilihatnya selama pelatihan[cite: 164].

---

### Exercise 5
**Q:** What makes Extra-Trees more random than regular Random Forests? How can this extra randomness help? Are Extra-Trees slower or faster?
**A:**
* [cite_start]**Lebih Acak**: Extra-Trees menggunakan *threshold* (ambang batas) acak untuk membelah fitur, sedangkan Random Forest mencari *threshold* optimal[cite: 227].
* **Manfaat**: Keacakan ekstra ini adalah bentuk *regularisasi* lain. [cite_start]Ini menukar bias yang sedikit lebih tinggi dengan variance yang lebih rendah[cite: 229].
* [cite_start]**Kecepatan**: Extra-Trees **lebih cepat** dilatih, karena mencari threshold optimal (yang dilakukan Random Forest) adalah salah satu tugas yang paling memakan waktu dalam menumbuhkan pohon[cite: 230].

---

### Exercise 6
**Q:** If your AdaBoost ensemble underfits the training data, which hyperparameters should you tweak and how?
**A:** Untuk mengurangi underfitting (meningkatkan kompleksitas):
1.  [cite_start]**`n_estimators`**: **Naikkan** jumlah estimator (pohon)[cite: 342].
2.  [cite_start]**`learning_rate`**: **Naikkan** learning rate (default 1.0)[cite: 313, 316].
3.  [cite_start]**Regularisasi Base Estimator**: **Kurangi** regularisasi pada base estimator (misalnya, jika menggunakan `DecisionTreeClassifier`, *naikkan* `max_depth` dari 1)[cite: 342].

---

### Exercise 7
**Q:** If your Gradient Boosting ensemble overfits the training set, should you increase or decrease the learning rate?
[cite_start]**A:** Anda harus **mengurangi** `learning_rate`[cite: 457, 458]. [cite_start]`learning_rate` yang lebih rendah (shrinkage) berarti setiap pohon memiliki kontribusi yang lebih kecil, yang meregularisasi model[cite: 458]. [cite_start]Namun, Anda mungkin perlu *menaikkan* `n_estimators` (menggunakan early stopping) untuk mengimbanginya[cite: 457, 481].

---

### Exercise 8 & 9
**Q:** (MNIST Stacking)
**A:** (Requires implementation)
* **Ex. [cite_start]8:** Latih beberapa classifier (Random Forest, Extra-Trees, SVM) di MNIST[cite: 631]. [cite_start]Gabungkan mereka menggunakan `VotingClassifier` (coba 'hard' dan 'soft')[cite: 632]. [cite_start]Evaluasi di validation set, lalu test set, dan bandingkan hasilnya[cite: 633, 634].
* **Ex. 9:** Gunakan classifier dari Ex. 8. [cite_start]Buat prediksi dari *validation set* untuk membuat *blending set* baru[cite: 635]. [cite_start]Latih "blender" (misalnya, `RandomForestClassifier` lain) pada blending set ini[cite: 636]. [cite_start]Evaluasi ensemble stacking ini di test set[cite: 637]. Bandingkan hasilnya dengan Voting Classifier dari Ex. [cite_start]8[cite: 638].

---

**Happy Ensembling! 🌲🌲🌲**