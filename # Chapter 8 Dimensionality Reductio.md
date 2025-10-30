# Chapter 8: Dimensionality Reduction

---

## 📖 Rangkuman Chapter 8

Chapter ini membahas **Dimensionality Reduction** (Pengurangan Dimensi), sebuah topik penting dalam Machine Learning. [cite_start]Banyak dataset di dunia nyata memiliki ribuan atau bahkan jutaan fitur (dimensi)[cite: 1678].

[cite_start]Jumlah fitur yang sangat besar ini tidak hanya membuat pelatihan menjadi sangat lambat, tetapi juga dapat mempersulit pencarian solusi yang baik[cite: 1679]. [cite_start]Masalah ini dikenal sebagai **"the curse of dimensionality"** (kutukan dimensi)[cite: 1680].

Untungnya, kita sering dapat mengurangi jumlah fitur secara signifikan. Chapter ini akan membahas "kutukan" tersebut dan dua pendekatan utama untuk mengatasinya (Proyeksi dan Manifold Learning), serta tiga teknik populer: **PCA**, **Kernel PCA**, dan **LLE**.

---

## 🎯 Topics Covered

| No | Topic | Description |
|----|-------|-------------|
| 1 | **The Curse of Dimensionality** | Mengapa data berdimensi tinggi itu aneh dan problematis |
| 2 | **Main Approaches** | Perbedaan antara Proyeksi (Projection) dan Manifold Learning |
| 3 | **PCA (Principal Component Analysis)** | Teknik reduksi dimensi paling populer (linier) |
| 4 | **Explained Variance** | Cara mengukur seberapa banyak informasi yang dipertahankan oleh PCA |
| 5 | **PCA Variants** | Randomized PCA (cepat) dan Incremental PCA (skala besar) |
| 6 | **Kernel PCA (kPCA)** | PCA versi nonlinier menggunakan "kernel trick" |
| 7 | **LLE (Locally Linear Embedding)** | Teknik Manifold Learning untuk "membuka" data yang terpilin |
| 8 | **Other Techniques** | Pengenalan singkat MDS, Isomap, t-SNE, dan LDA |

---

## 🌌 The Curse of Dimensionality

[cite_start]Kita terbiasa hidup di ruang 3D, sehingga intuisi kita gagal saat membayangkan ruang berdimensi tinggi[cite: 1698].

**Masalah Utama:**
* **Data Menjadi Jarang (Sparse):** Di ruang berdimensi tinggi, "ruang" sangatlah besar. [cite_start]Dua titik acak dalam sebuah *unit square* (1x1) rata-rata berjarak 0.52, namun di *unit hypercube* 1.000.000 dimensi, jarak rata-ratanya sekitar 408.25![cite: 1718, 1720, 1721].
* [cite_start]**Semua Titik Ada di "Tepi":** Di hypercube 10.000 dimensi, probabilitas sebuah titik acak berada "sangat dekat dengan batas" adalah > 99.999999%[cite: 1711].
* [cite_start]**Risiko Overfitting:** Karena data sangat jarang, instance baru kemungkinan besar akan jauh dari instance pelatihan mana pun[cite: 1723]. [cite_start]Prediksi menjadi tidak dapat diandalkan (ekstrapolasi besar), yang meningkatkan risiko overfitting[cite: 1724, 1725].
* [cite_start]**Tidak Dapat Diatasi dengan Data:** Jumlah data yang diperlukan untuk menjaga kepadatan (density) data tumbuh secara **eksponensial** dengan jumlah dimensi[cite: 1727].

---

## 💡 Main Approaches

Ada dua pendekatan utama untuk mengurangi dimensi:

| Approach | Description | Analogi | Kapan Digunakan |
|---|---|---|---|
| **Projection** | [cite_start]Memproyeksikan data ke *subspace* berdimensi lebih rendah (hyperplane)[cite: 1731, 1758]. | Membuat bayangan 2D dari objek 3D di atas lantai. | [cite_start]Ketika data "tergeletak" di dekat subspace yang "datar" (linier)[cite: 1734]. |
| **Manifold Learning** | [cite_start]Mengasumsikan data terletak pada *manifold* (bentuk) berdimensi rendah yang terpilin atau tertekuk di dalam ruang berdimensi tinggi [cite: 1810-1811, 1815-1816]. | [cite_start]"Membuka" gulungan Swiss roll (kue gulung) untuk menjadikannya lembaran 2D datar[cite: 1786]. | [cite_start]Ketika data memiliki struktur nonlinier yang terpilin (seperti Swiss roll)[cite: 1770]. |

---

## ⚙️ PCA (Principal Component Analysis)

[cite_start]**PCA** adalah algoritme reduksi dimensi yang paling populer[cite: 1881]. [cite_start]PCA bekerja dengan cara **Proyeksi**[cite: 1882].

### Core Idea: Preserving the Variance

Bagaimana PCA memilih hyperplane (subspace) terbaik untuk proyeksi?
[cite_start]Ia memilih hyperplane yang **mempertahankan jumlah variance (varians) semaksimal mungkin**[cite: 1887, 1905].


[cite_start]Ini setara dengan memilih sumbu yang **meminimalkan mean squared distance** (jarak kuadrat rata-rata) antara data asli dan proyeksinya (ini disebut *reconstruction error*)[cite: 1906].

### Principal Components (PCs)

PCA menemukan sumbu-sumbu yang memaksimalkan varians:
* [cite_start]**1st PC**: Sumbu yang menyimpan varians terbesar[cite: 1909].
* [cite_start]**2nd PC**: Sumbu kedua, yang **ortogonal (tegak lurus)** terhadap PC pertama, yang menyimpan varians *sisa* terbesar[cite: 1910].
* [cite_start]**3rd PC**: Ortogonal terhadap dua PC pertama, menyimpan varians sisa terbesar, dan seterusnya[cite: 1912].

[cite_start]Sumbu-sumbu ini (unit vector $c_1, c_2, ...$) disebut **Principal Components** (Komponen Utama)[cite: 1913].

### Math (SVD)

[cite_start]Untuk menemukan semua Principal Component, PCA menggunakan teknik dekomposisi matriks standar yang disebut **Singular Value Decomposition (SVD)** pada data yang sudah di-center (mean=0)[cite: 1924, 1933].

---

## 📊 Using PCA

### Scikit-Learn

[cite_start]Scikit-Learn menangani PCA dengan mudah dan otomatis melakukan centering data[cite: 1950, 1954].

```python
from sklearn.decomposition import PCA

# Mengurangi dimensi menjadi 2
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```

### Explained Variance Ratio

[cite_start]Setelah di-fit, Anda bisa memeriksa berapa banyak varians yang ditangkap oleh setiap PC menggunakan `explained_variance_ratio_`[cite: 1957].

```python
>>> pca.explained_variance_ratio_
аггау([0.84248607, 0.14631839])
```
[cite_start]*Artinya: PC pertama menangkap 84.2% varians, dan PC kedua menangkap 14.6% varians[cite: 1962].*

### Choosing the Right Number of Dimensions

[cite_start]Daripada memilih `n_components` secara acak, lebih baik memilih jumlah dimensi yang mempertahankan sebagian besar varians (misalnya, 95%)[cite: 1965].

**Opsi 1: Set rasio varians**
Scikit-Learn memungkinkan Anda mengatur `n_components` sebagai float antara 0.0 dan 1.0:

```python
# Otomatis memilih jumlah dimensi untuk 95% varians
pca = PCA(n_components=0.95)
[cite_start]X_reduced = pca.fit_transform(X_train) [cite: 1973-1974]
```

**Opsi 2: Plot "Elbow"**
Anda dapat memplot jumlah varians kumulatif vs. jumlah dimensi. [cite_start]Biasanya akan ada "siku" (elbow) di mana penambahan dimensi baru tidak lagi memberikan banyak varians tambahan [cite: 1975-1976].


### PCA for Compression & Reconstruction

[cite_start]PCA sangat berguna untuk mengompresi dataset[cite: 1991]. [cite_start]Misalnya, dataset MNIST (784 fitur) dapat dikompresi hingga 95% variansnya hanya dengan ~154 fitur[cite: 1993, 1999].

[cite_start]Dataset yang terkompresi dapat "didekompresi" kembali ke dimensi aslinya menggunakan `inverse_transform()`[cite: 1996].

```python
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
# Mengembalikan ke 784 dimensi (dengan sedikit kehilangan info)
[cite_start]X_recovered = pca.inverse_transform(X_reduced) [cite: 1999-2004]
```
[cite_start]Transformasi balik ini tidak sempurna; ia kehilangan informasi (misalnya, 5% varians yang kita buang), dan ini disebut **reconstruction error** [cite: 1997-1998].


---

## 🗂️ PCA Variants

| Variant | Class | Kapan Digunakan |
|---|---|---|
| **Randomized PCA** | `PCA(svd_solver="randomized")` | [cite_start]**Untuk percepatan.** Algoritme stokastik yang menemukan *perkiraan* PC dengan cepat[cite: 2023]. [cite_start]Jauh lebih cepat jika $d$ (dimensi target) jauh lebih kecil dari $n$ (fitur asli)[cite: 2024]. |
| **Incremental PCA** | `IncrementalPCA()` | [cite_start]**Untuk dataset sangat besar** (out-of-core) atau data streaming[cite: 2030, 2032]. [cite_start]Model dilatih dalam *mini-batches* menggunakan metode `partial_fit()`[cite: 2030, 2034]. |

---

## 🌀 Kernel PCA (kPCA)

[cite_start]Ini adalah adaptasi dari PCA menggunakan **"kernel trick"** (seperti di Bab 5)[cite: 2049, 2051].

[cite_start]**Ide Dasar:** kPCA memungkinkan **proyeksi nonlinier**[cite: 2051]. [cite_start]Ini secara implisit memetakan data ke ruang fitur berdimensi sangat tinggi, di mana pemisahan linier (PCA) kemudian dapat diterapkan, yang setara dengan pemisahan nonlinier di ruang asli[cite: 2050].

[cite_start]Ini sangat baik untuk **membuka manifold** (seperti Swiss roll) atau **mempertahankan cluster** setelah proyeksi[cite: 2056].

```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
[cite_start]X_reduced = rbf_pca.fit_transform(X) [cite: 2059-2061]
```

### Selecting Kernel & Tuning

kPCA adalah unsupervised, jadi tidak ada metrik yang jelas. Ada dua cara untuk memilih kernel/hyperparameter:

1.  **Grid Search (Supervised):** Buat `Pipeline` yang berisi kPCA diikuti oleh classifier (misalnya, `LogisticRegression`). [cite_start]Gunakan `GridSearchCV` untuk menemukan parameter kPCA (seperti `kernel` dan `gamma`) yang menghasilkan **akurasi klasifikasi terbaik** di akhir pipeline [cite: 2092-2094, 2101-2102].
2.  [cite_start]**Reconstruction Pre-image (Unsupervised):** Menemukan kernel yang meminimalkan *reconstruction pre-image error*[cite: 2112, 2120].
    * [cite_start]Rekonstruksi di kPCA rumit karena feature space bisa tak terhingga[cite: 2117].
    * [cite_start]Kita mencari "pre-image": titik di ruang asli yang paling dekat dengan titik rekonstruksi [cite: 2118-2119].
    * [cite_start]Atur `fit_inverse_transform=True` di `KernelPCA` (ini akan melatih model regresi di latar belakang) untuk mengaktifkan `inverse_transform()` [cite: 2137-2138, 2143].

---

## 🗺️ LLE (Locally Linear Embedding)

[cite_start]LLE adalah teknik **Manifold Learning** nonlinier yang kuat, dan **bukan** berbasis proyeksi [cite: 2151-2152].

**Cara Kerja:**
[cite_start]LLE bekerja dengan mengukur bagaimana setiap instance berhubungan secara linier dengan tetangga terdekatnya (c.n. / closest neighbors)[cite: 2153].
1.  [cite_start]**Step 1: Identify Neighbors.** Untuk setiap instance $x^{(i)}$, temukan $k$ tetangga terdekatnya[cite: 2180].
2.  [cite_start]**Step 2: Find Local Weights.** Temukan bobot $w_{i,j}$ yang paling baik merekonstruksi $x^{(i)}$ sebagai kombinasi linier dari tetangganya ($\sum w_{i,j} x^{(j)}$) [cite: 2180-2181]. [cite_start]Bobot $W$ ini menyimpan "hubungan lokal"[cite: 2188].
3.  [cite_start]**Step 3: Map to Low-D.** Cari representasi $z^{(i)}$ berdimensi rendah (misalnya, 2D) di mana hubungan lokal yang *sama* (menggunakan bobot $W$ yang *sama*) paling baik dipertahankan [cite: 2189, 2192-2193].

[cite_start]LLE sangat baik dalam "membuka" manifold yang terpilin[cite: 2154].

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
[cite_start]X_reduced = lle.fit_transform(X) [cite: 2157-2159]
```

[cite_start]**Kelemahan:** Kompleksitas komputasi $O(dm^2)$ pada langkah terakhir membuatnya **tidak berskala baik** untuk dataset yang sangat besar (m = jumlah instance)[cite: 2203].

---

## 📚 Other Dimensionality Reduction Techniques

| Technique | Description |
|---|---|
| **MDS** | [cite_start]**Multidimensional Scaling:** Mengurangi dimensi sambil mencoba **mempertahankan jarak** antar instance[cite: 2213]. |
| **Isomap** | [cite_start]**Isomap:** Membuat graf tetangga, lalu mengurangi dimensi sambil mempertahankan **jarak geodesik** (jarak "berjalan di sepanjang" manifold)[cite: 2216]. |
| **t-SNE** | [cite_start]**t-Distributed Stochastic Neighbor Embedding:** Menjaga instance serupa tetap dekat dan instance tidak serupa tetap jauh[cite: 2218]. [cite_start]Sangat baik untuk **visualisasi cluster** (misalnya, MNIST)[cite: 2219]. |
| **LDA** | **Linear Discriminant Analysis:** Sebenarnya adalah algoritme klasifikasi. [cite_start]Ia mempelajari sumbu-sumbu yang paling **memisahkan antar kelas**, dan sumbu-sumbu ini dapat digunakan untuk reduksi dimensi [cite: 2221-2222]. |

---

## 🔧 Exercises (from the book)

### Exercise 1
**Q:** What are the main motivations for reducing a dataset's dimensionality? What are the main drawbacks?
**A:**
* [cite_start]**Motivasi:** Terutama untuk **mempercepat pelatihan** (mengubah masalah yang tidak dapat ditangani menjadi dapat ditangani)[cite: 1679, 1681, 1689]. [cite_start]Juga sangat berguna untuk **visualisasi data** (Data Viz) dengan menguranginya menjadi 2D atau 3D [cite: 1690-1691].
* [cite_start]**Kelemahan:** Menyebabkan **kehilangan informasi** (seperti kompresi JPEG)[cite: 1685]. [cite_start]Dapat menurunkan performa model[cite: 1685]. [cite_start]Membuat pipeline model menjadi lebih kompleks dan lebih sulit dirawat[cite: 1686].

---

### Exercise 2
**Q:** What is the curse of dimensionality?
**A:** Ini adalah fakta bahwa banyak hal berperilaku sangat berbeda di ruang berdimensi tinggi. [cite_start]Data menjadi sangat **jarang (sparse)**; instance pelatihan cenderung sangat **jauh satu sama lain**[cite: 1723]. [cite_start]Ini membuat prediksi menjadi tidak dapat diandalkan (berdasarkan ekstrapolasi besar) dan meningkatkan risiko **overfitting** [cite: 1724-1725].

---

### Exercise 3
**Q:** Once a dataset's dimensionality has been reduced, is it possible to reverse the operation?
**A:** **Tergantung.**
* **PCA:** Ya, dapat dibalikkan menggunakan metode `inverse_transform()`. Namun, ini adalah "rekonstruksi" yang **lossy** (ada kehilangan data); [cite_start]Anda tidak mendapatkan data asli kembali karena Anda membuang varians [cite: 1996-1998].
* **kPCA:** Ya, tetapi lebih rumit. Ini disebut menemukan "pre-image". [cite_start]Anda dapat mengaktifkannya dengan `fit_inverse_transform=True` [cite: 2118-2119, 2137-2138].
* **LLE:** (Teks tidak menyebutkan metode inversi langsung).

---

### Exercise 4
**Q:** Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?
**A:** **Ya, bisa,** tapi PCA adalah algoritme linier yang akan mencoba "meratakan" data. [cite_start]Ini akan efektif dalam mengurangi dimensi tetapi akan kehilangan semua struktur nonlinier yang kompleks (misalnya, "menghancurkan" Swiss roll menjadi satu bidang datar)[cite: 1785]. [cite_start]Untuk dataset nonlinier, **kPCA** [cite: 2051] [cite_start]atau **LLE** [cite: 2154] adalah pilihan yang jauh lebih baik.

---

### Exercise 5
**Q:** Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?
**A:** **Tidak mungkin diketahui** tanpa melihat datasetnya. Jumlah dimensi akan bergantung pada struktur internal data tersebut. [cite_start]Jika datasetnya MNIST (dari 784D), hasilnya adalah 154D[cite: 1999]. [cite_start]Jika dataset lain, bisa jadi 100D[cite: 1977], atau 500D. Jawabannya **bergantung pada dataset**.

---

### Exercise 6
**Q:** In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?
**A:**
* **Vanilla PCA (Full SVD):** Pilihan default untuk dataset yang muat di memori.
* [cite_start]**Incremental PCA (IPCA):** Ketika dataset **terlalu besar** untuk muat di memori (out-of-core) atau untuk **data streaming (online)**[cite: 2030, 2032].
* [cite_start]**Randomized PCA:** Ketika $d$ (dimensi target) jauh lebih kecil dari $n$ (fitur asli) dan Anda membutuhkan **performa yang jauh lebih cepat**[cite: 2024].
* [cite_start]**Kernel PCA (kPCA):** Ketika dataset **nonlinier** (misalnya, terpilin seperti Swiss roll)[cite: 2051, 2056].

---

### Exercise 7
**Q:** How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?
**A:**
1.  **Kualitas Pipeline (Supervised):** Ukur performa dari *tugas downstream* (misalnya, klasifikasi). [cite_start]Reduksi dimensi yang baik adalah yang meningkatkan akurasi atau sangat mempercepat pelatihan tanpa terlalu merusak akurasi akhir[cite: 2092].
2.  [cite_start]**Reconstruction Error (Unsupervised):** Ukur "reconstruction error" (atau "pre-image error" untuk kPCA)[cite: 1998, 2112]. Reduksi dimensi yang baik memiliki error rekonstruksi yang rendah.

---

### Exercise 8
**Q:** Does it make any sense to chain two different dimensionality reduction algorithms?
**A:** **Ya.** (Teks tidak secara eksplisit membahas ini, tetapi ini adalah praktik umum). Misalnya, Anda dapat menggunakan PCA untuk menghilangkan redundansi linier dengan cepat, lalu menggunakan t-SNE atau LLE pada data yang sudah dikurangi tersebut untuk memvisualisasikan struktur manifold nonliniernya.

---

### Exercise 9 & 10
**Q:** (MNIST Exercises)
**A:** (Requires implementation)
* **Ex. 9:** Latih Random Forest pada MNIST. Lalu gunakan PCA (95% variance). Latih Random Forest lagi pada data tereduksi. [cite_start]Bandingkan waktu dan akurasi [cite: 2270-2274].
* **Ex. 10:** Gunakan t-SNE untuk mengurangi MNIST menjadi 2D dan plot hasilnya. [cite_start]Bandingkan visualisasi dengan PCA, LLE, atau MDS [cite: 2275-2279].

---

**Happy Reducing! 📉**