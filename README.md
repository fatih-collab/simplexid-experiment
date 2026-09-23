# SimplEx-ID — Kerangka Kode Eksperimen

Kerangka kode ini mengimplementasikan pipeline SimplEx-ID (R0–R8 + formula skor)
dengan flag on/off per mekanisme, sehingga varian ablation V0–V9 (lihat
`Rancangan_Eksperimen_SimplEx-ID.docx`) tinggal dikonfigurasi lewat `config.py`
tanpa duplikasi kode.

## Instalasi

```bash
pip install -r requirements.txt
python -m spacy_udpipe download id      # model UD Bahasa Indonesia
```

## Struktur

```
src/
  config.py          # definisi 10 varian V0-V9 sebagai kombinasi flag
  data_loader.py      # load TermA & HoASA dari HuggingFace datasets
  parser.py            # wrapper UDPipe -> struktur token+dependensi
  dam.py                # kamus Domain-Aspect Mapping + disambiguasi kontekstual
  rules.py              # R0,R1,R2,R5,R6,R7 (eksplisit) + R3,R4 (ekspansi konjungtif)
  scoring.py            # Neg/Int (statis maupun berbobot-jarak) + scope consistency
  pipeline.py           # orkestrasi: parse -> rules -> expand -> R8 -> scoring
  evaluate.py           # P/R/F1/Accuracy vs ground truth TermA
  secondary_tagging.py  # deteksi semi-otomatis kalimat COORD/IMPLICIT_AMBIG/ASYM_MOD
  run_experiment.py     # loop V0-V9, k-fold CV, paired t-test -> results/
```

## Alur kerja yang disarankan

1. `python src/data_loader.py` — unduh & cache TermA/HoASA, cetak statistik dasar.
2. `python src/secondary_tagging.py` — beri tag kandidat COORD/IMPLICIT_AMBIG/ASYM_MOD
   secara semi-otomatis (lihat catatan di bagian bawah) sebagai titik awal anotasi manual.
3. Lengkapi `src/dam.py` — isi kamus DAM domain hotel (starter kecil sudah disediakan,
   perlu diperluas dari hasil eksplorasi TermA/HoASA seperti dijelaskan di
   Bagian 4 rancangan eksperimen).
4. `python src/run_experiment.py` — menjalankan seluruh varian V0-V9 pada test set,
   menghasilkan `results/metrics.csv` dan `results/significance.csv`.

## Catatan penting

- Kode ini adalah **kerangka kerja (scaffold)**, bukan implementasi final —
  beberapa bagian (terutama `dam.py` dan bobot kemiripan pada `dam.py`'s
  `disambiguate()`) memerlukan Anda mengisi data domain-spesifik.
- `secondary_tagging.py` mendeteksi kandidat kasus secara heuristik (mis. relasi
  `conj` untuk COORD) — ini MEMBANTU mempercepat anotasi manual, bukan pengganti
  anotasi manual itu sendiri (lihat Bagian 3 rancangan eksperimen soal kappa
  inter-annotator).
- Belum diuji end-to-end di lingkungan ini karena keterbatasan akses jaringan
  (tidak bisa mengunduh model UDPipe/dataset HuggingFace dari sandbox ini).
  Jalankan dan debug di lingkungan lokal Anda.
