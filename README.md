# SimplEx-ID — Kerangka Kode Eksperimen

Kerangka kode ini mengimplementasikan pipeline SimplEx-ID (R0–R8 + formula skor)
dengan flag on/off per mekanisme, sehingga varian ablation V0–V9 (lihat
`Rancangan_Eksperimen_SimplEx-ID.docx`) tinggal dikonfigurasi lewat `config.py`
tanpa duplikasi kode.

## Instalasi

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Model UDPipe tidak disimpan di git. Unduh sesuai `resources/models/udpipe/SOURCE.md`
dan verifikasi checksum-nya. Jangan memakai `spacy_udpipe.download("id")`: URL lamanya
mati dan menghasilkan file HTML.

```bash
python -m pytest -q          # semua test harus lulus
```

## Struktur

```
src/simplexid/
  config.py             # definisi varian ablation sebagai kombinasi flag
  data.py               # loader & preprocessing TermA/HoASA (H1)
  gold.py               # pembuatan & validasi gold data (H2)
  parser.py             # UDPipe (ufal.udpipe) -> Token/ParsedSentence, CoNLL-U
  dam.py                # kamus Domain-Aspect Mapping + disambiguasi kontekstual
  rules.py              # R0,R1,R2,R5,R6,R7 (eksplisit) + R3,R4 (ekspansi konjungtif)
  scoring.py            # Neg/Int (statis maupun berbobot-jarak) + scope consistency
  pipeline.py           # orkestrasi: parse -> rules -> expand -> R8 -> scoring
  secondary_tagging.py  # kandidat COORD/IMPLICIT_AMBIG/ASYM_MOD
experiments/run_h2.py   # runner gold data (H2)
resources/models/udpipe # model parser + SOURCE.md (provenance, checksum, lisensi)
reports/research_log.md # catatan tiap fase
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
