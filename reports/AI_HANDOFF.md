# SimplEx-ID — Experiment Design Log & AI Handoff

> **Status per 2026-11-05.** Repo: `github.com/fatih-collab/simplexid-experiment`, branch `main`. H4.0–H4.3 selesai di lokal; lihat §0A. Bagian lain ditulis 2026-10-19; jika bertentangan, §0A dan `reports/research_log.md` yang benar.
> Dokumen ini adalah konteks lengkap untuk asisten AI yang membantu pemilik eksperimen.
> Jika ada yang bertentangan dengan isi repo terbaru, **repo yang benar**; minta pengguna mengirim file terbaru.

---

## 0. Instruksi untuk AI yang membaca dokumen ini

**Peranmu:** mitra Data Scientist / NLP researcher yang membantu pemilik eksperimen (peneliti, berkomunikasi dalam Bahasa Indonesia) untuk:

1. **Merancang langkah lanjutan** (fase H4–H7, lihat §10) — memberi rekomendasi konkret, bukan survei opsi.
2. **Membantu review gold data** (fase H5, lihat §11) — menjelaskan aturan anotasi, membantu memutuskan kasus sulit, merancang prosedur verifikasi & inter-annotator agreement.

**Aturan kerja:**
- Baca §7 (keputusan yang sudah diambil) dan §8 (aturan integritas) sebelum memberi saran. Jangan mengusulkan ulang hal yang sudah diputuskan kecuali ada bukti baru, dan sebutkan buktinya.
- **Jangan pernah** menyarankan tuning, pemilihan rule, pemilihan leksikon, atau analisis error memakai *final test* (TermA validation). Semua keputusan desain hanya dari *development* (TermA train).
- Saat membantu review gold, **kamu memberi saran; manusia yang memutuskan.** Tujuan H5 justru memverifikasi label secara manusiawi, karena sebagian label awal kemungkinan dibuat dengan bantuan AI. Jangan mengisi label massal tanpa diperiksa pengguna, dan tandai keraguanmu.
- Jika butuh data (baris CSV, angka, isi file), minta pengguna menempelkannya. Jangan mengarang angka.
- Eksperimen ini ditujukan untuk **paten** dan **artikel ilmiah**: utamakan reproducibility, kejujuran pelaporan, dan pemisahan dev/final.

---

## 0A. Pembaruan 2026-11-05 (baca dulu)

- **Risiko paten: repo GitHub ini publik**, dan kode semua mekanisme sudah terbuka sejak 2026-09-23. Pengguna memutuskan repo tetap publik dan push dilanjutkan; perlu penilaian konsultan HKI.
- **Varian kini A0–A6, L_*, S_R4** (`config.py`), menggantikan V0–V9 di §2.4. Pemetaan: V0=A0, V1=A1, V2=S_R4, V3=A2, V4=A3, V5=A4, V7=A5, V9=A6.
- **Leksikon: `merged_v1`** (SentiStrengthID + ID-OpinionWords), file tidak ada di git. D2 (leksikon domain) belum diputuskan.
- **Head F1 dev A2: 0,371 → 0,588** (P 0,617, R 0,561). Tanpa kosakata aspek dari anotasi train: 0,478. Sistem penuh **tidak** bebas data beranotasi.
- **Polaritas oracle-pair dev: 0,75.**
- **Bukti per mekanisme di dev:** R4 signifikan (+0,017; +0,093 di COORD_OPINION). R3 hanya di COORD_ASPECT (+0,058). **R8, R8 kontekstual, DW, SC belum punya bukti.**
- **Jangkauan di 3.000 kalimat dev:** SC mengubah 44 label (0,8%), DW 4 label + 407 magnitudo, R8 kontekstual 7 dari 209. Gold polaritas 400 item tidak bisa menguji SC; perlu anotasi tertarget.
- **Lembar SC tertarget:** `data/annotation/sc_targeted/` (44 pasangan, menunggu anotasi).
- **Review:** `gold_dev_auto_1a1o.csv` di repo masih 468 AUTO. Gold dev polaritas (14) dan implisit (195) belum diverifikasi.
- **Belum dikerjakan:** H4.4 (R8), H4.5 (tuning), H4.6 (baseline eksternal).

---

## 1. Ringkasan proyek

**SimplEx-ID** adalah sistem *Aspect-Based Sentiment Analysis* (ABSA) **berbasis aturan** untuk ulasan hotel berbahasa Indonesia. Sistem ini tidak memakai model yang dilatih. Inputnya kalimat ulasan, outputnya pasangan *(aspek, opini, skor sentimen)*. Setiap pasangan bisa dilacak ke aturan yang menghasilkannya (explainable).

**Tujuan akhir:**
- **Paten** atas metode (dalam proses; klaim akan didiskusikan dengan konsultan HKI). Bukti "efek teknis" diambil dari studi ablation.
- **Artikel ilmiah**. Kontribusi terkuat yang realistis:
  - gold data baru di atas TermA (pemasangan aspek–opini, polaritas, aspek implisit);
  - mekanisme *scope consistency*;
  - evaluasi yang ketat (oracle upper bound, uji signifikansi).

**Empat mekanisme yang ingin diklaim:**

| Mekanisme | Isi | Catatan novelty |
|---|---|---|
| **R3/R4** ekspansi konjungtif | R3: "kecepatan **dan** stabilitas baik" → 2 aspek. R4: "lemah **dan** tidak stabil" → 2 opini | Mirip *Double Propagation* (Qiu et al., 2011), jadi harus disitasi dan dibandingkan |
| **R8** aspek implisit | Kamus Domain-Aspect Mapping (DAM) + disambiguasi kontekstual ("mahal" → kategori sesuai konteks) | Mirip Liu 2005 / Fei 2012 |
| **DW scoring** | Faktor negasi/intensifier meluruh menurut **jarak dependensi** ke kata opini | Bandingkan dengan SO-CAL (Taboada et al., 2011) |
| **Scope consistency (SC)** | Pasangan hasil R3/R4 menghitung negasi/intensifier **sendiri**, tidak mewarisi dari pasangan asal | Kemungkinan paling orisinal |

**Urutan publikasi:** permohonan paten diajukan **sebelum** publikasi apa pun. UU Paten No. 13/2016 Pasal 6 memberi masa tenggang 6 bulan, tetapi banyak negara tidak. Klaim sebaiknya *lexicon-agnostic*.

---

## 2. Arsitektur sistem

### 2.1 Pipeline (`src/simplexid/pipeline.py`)

```
token TermA ─► UDPipe (cache CoNLL-U) ─► R0,R1,R2,R5,R6,R7 (eksplisit)
           ─► [R3] [R4] ekspansi konjungtif ─► skor Neg × Int × Lexicon per pasangan
           ─► [R8] opini "yatim" → kategori aspek implisit via DAM
           ─► [scope consistency] hitung ulang faktor untuk pasangan R3/R4
```

### 2.2 Rules (`rules.py`). Semua bekerja pada **head token**, bukan frasa

| Rule | Relasi UD | Kondisi |
|---|---|---|
| R0 | `amod` | opini ADJ/VERB memodifikasi aspek NOUN |
| R1 | `nsubj` | aspek NOUN adalah subjek opini |
| R2 | `obj` | opini VERB, objek NOUN |
| R5 | `obl` | opini terhubung ke NOUN lewat oblique |
| R6 | `compound` | menggabungkan teks aspek majemuk (tidak menambah pasangan) |
| R7 | `relcl` | **tidak pernah aktif**: parser memakai label `acl` |
| R3 | `conj` antar NOUN | tambah aspek |
| R4 | `conj` antar ADJ | tambah opini |
| R8 | orphan ADJ/VERB + DAM | aspek implisit (kategori HoASA) |

### 2.3 Skor (`scoring.py`)

- `score = Neg × Int × lexicon(lemma_opini)`.
- Neg bernilai −1 jika ada kata negasi (`advmod`) pada opini. Pada mode DW, hanya jika jaraknya ≤ `NEG_DISTANCE_THRESHOLD`=3.
- Int: amplifier 1,8 / downtoner 0,6. Pada mode DW: `1 + (base−1)·exp(−λ·d)`, λ=0,4.
- Polaritas: skor > 0,05 → POSITIVE, < −0,05 → NEGATIVE, selain itu NEUTRAL.
- Threshold dan λ **belum di-tuning**.

### 2.4 Varian ablation saat ini (`config.py`): **cacat dan akan dirancang ulang**

| Varian | Flag aktif |
|---|---|
| V0 | baseline R0–R7 |
| V1 | +R3 |
| V2 | +R4 |
| V3 | +R3+R4 |
| V4 | V3+R8 statis |
| V5 | V3+R8 kontekstual |
| V6 | = V5 (**identik**) |
| V7 | V5+DW |
| V8 | = V7 (**identik**) |
| V9 | V7+SC |

Cacat yang diketahui:
- V5≡V6 dan V7≡V8 secara flag.
- V8 vs V9 **selalu nol**, karena pipeline sudah menghitung faktor per pasangan sehingga baseline "pewarisan" tidak pernah terjadi.
- V4 vs V5 nol di gold implisit, karena kalimat implisit tidak punya aspek eksplisit sebagai konteks dan TermA tidak dikelompokkan per ulasan.

---

## 3. Data dan split

| Dataset | Isi | Pemakaian |
|---|---|---|
| **TermA** (IndoNLU, AiryRooms) | 5.000 kalimat. Label IOB span ASPECT & SENTIMENT. **Tidak ada** pemasangan dan polaritas. | Ground truth utama |
| **HoASA** (IndoNLU) | 2.854 ulasan, 10 kategori aspek hotel | Ontologi kategori implisit / DAM |

**Kebijakan split (dikunci):**

| Split | Sumber | Jumlah | Pemakaian |
|---|---|---|---|
| development | TermA train | 3.000 | Pengembangan rules, tuning, error analysis |
| **final** | TermA validation | 1.000 | **TERKUNCI**: hanya untuk satu kali run final (H6) |
| official test | TermA test | 1.000 | Label di-mask, tidak dipakai untuk metrik |

Random seed 42.

---

## 4. Gold data (hasil H2)

Lokasi `data/gold/`. Semua CSV memakai pemisah **koma**, encoding UTF-8-sig.

| File | Isi | Baris | Status |
|---|---|---|---|
| `gold_review_log.csv` | Keputusan pemasangan per kalimat (`accepted_pairs`), **sumber kebenaran** | 1.314 kalimat | REVIEWED (kemungkinan berbantuan AI) |
| `gold_pairs.csv` | Pasangan eksplisit, **dibuat otomatis** dari review log. Jangan diedit manual. | 3.202 pasangan | PASS validasi |
| `gold_polarity.csv` | Polaritas per pasangan, subset terstratifikasi | 1.000 (600 final + 400 dev) | REVIEWED, ada `confidence` di `notes` |
| `gold_implicit.csv` | Aspek implisit per opini "yatim" | 451 (201 final + 250 dev) | REVIEWED, ada `confidence` di `notes` |
| `gold_dev_auto_1a1o.csv` | 468 kalimat dev 1-aspek-1-opini, pre-filled `A1-O1` | 468 | **AUTO**, sedang diverifikasi pengguna |

### 4.1 Skema dan contoh

**gold_review_log.csv.** `aspects` / `opinions` berisi daftar span bernomor dari TermA. `accepted_pairs` menghubungkannya.

```
sentence_id=terma_train_00023 | sentence="nyaman banget kamarnya . bersiih . dan buat betah . ..."
aspects="A1=kamarnya" | opinions="O1=nyaman banget | O2=bersiih"
accepted_pairs="A1-O1|A1-O2" | complex_type=MULTI_OPINION | review_scope=DEV_COMPLEX | annotation_status=REVIEWED
```

Nilai `accepted_pairs`:
- `A<i>-O<j>` dipisah `|`;
- `NONE`: ada span, tetapi tidak ada pasangan yang valid;
- `AMBIGUOUS`: kalimat dikeluarkan dari evaluasi.

`review_scope`: FINAL_COMPLEX 646, FINAL_SIMPLE 176, DEV_COMPLEX 440, DEV_SIMPLE_AUDIT 52.

**gold_pairs.csv:** `pair_id=terma_train_00010__A1__O2`, aspect "kamar" [4,4], opinion "banyak semut" [5,6], `relation_pattern` (ONE_TO_ONE / ONE_TO_MANY / MANY_TO_ONE / MANY_TO_MANY). Kolom `relation_type` dan `secondary_tags` **masih kosong**; akan diisi di H5.

**gold_polarity.csv:** `gold_polarity` ∈ {POSITIVE, NEGATIVE, NEUTRAL}.
- `polarity_stratum`: NEGATION 282, INTENSIFIER 263, STRUCTURAL_MULTI 267, NORMAL 188.
- Distribusi label: POSITIVE 605, NEGATIVE 384, NEUTRAL 11.

```
pair_id=terma_validation_00418__A6__O5 | aspect="bed covernya" | opinion="sempat bau pesing"
gold_polarity=NEGATIVE | notes="confidence=HIGH|negative lexical cue"
```

**gold_implicit.csv:**
- `decision` ∈ {IMPLICIT 407, NOT_IMPLICIT 42, AMBIGUOUS 2}.
- `gold_aspect_category` ∈ 10 kategori HoASA. Isi saat ini: general 312, kebersihan 68, service 24, bau 3.
- Ada juga `gold_polarity`.

```
sentence="paling nyaman dari yang prnah saya inap ." | opinion="nyaman" [1,1]
decision=IMPLICIT | gold_aspect_category=general | gold_polarity=POSITIVE | notes="confidence=MEDIUM|general/domain-level sentiment"
```

**gold_dev_auto_1a1o.csv:**

```
sentence="pelayanan nya sangat ramah ." | aspects="A1=pelayanan nya" | opinions="O1=sangat ramah"
accepted_pairs=A1-O1 | review_scope=DEV_AUTO_1A1O | annotation_status=AUTO
```

### 4.2 Sebaran confidence (kolom `notes`)

| File | Scope | HIGH | MEDIUM | LOW |
|---|---|---|---|---|
| gold_polarity | DEV | 386 | 14 | 0 |
| gold_polarity | FINAL | 583 | 17 | 0 |
| gold_implicit | DEV | 55 | 194 | 1 |
| gold_implicit | FINAL | 40 | 160 | 1 |

### 4.3 Universe evaluasi (dihitung oleh `evaluation/gold_loader.py`)

| Split | Kalimat | Pasangan | Komposisi |
|---|---|---|---|
| final | 1.000 | 2.077 | 822 reviewed + 178 tanpa aspek (gold kosong) |
| dev | 1.478 | 1.593 | 492 reviewed + 468 AUTO_1A1O + 518 tanpa span |

Di dev, **1.522 kalimat kompleks belum direview** dan dikeluarkan dari evaluasi. Komposisi kalimat kompleks: dev 46%, final 79%. Akibatnya skor dev cenderung lebih tinggi daripada final.

---

## 5. Infrastruktur yang sudah dibangun (H3.1–H3.3)

Lingkungan: Windows 11, Python 3.14.3. Versi dipin di `requirements.txt` (ufal.udpipe 1.4.0.1, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, datasets 5.0.1). Ada **53 test**, semua lulus.

| Komponen | File | Catatan |
|---|---|---|
| Parser | `src/simplexid/parser.py` | `ufal.udpipe` langsung. spacy-udpipe dibuang karena tidak pernah mem-parse input pretokenized. Model `indonesian-gsd-ud-2.5-191206.udpipe` (MD5 `e0167ecc…`, lisensi **CC BY-NC-SA 4.0, non-komersial**). |
| Cache parse | `parse_cache.py`, `data/cache/parses/` | 5.000 kalimat, tepat 1 pohon per kalimat. Manifest berisi hash model, input, dan cache. |
| Oracle | `oracle.py` | head span sintaksis, jalur dependensi aspek→opini |
| Evaluasi | `evaluation/` | Matching strict/**head**/overlap one-to-one. Polaritas end-to-end / conditional / **oracle-pair**. Implisit vs baseline mayoritas. Paired bootstrap 10k, McNemar exact, Holm. |
| Provenance | `evaluation/run_record.py` | Manifest per run (commit git, dirty, hash kode/resource/gold). **Final terkunci** sampai `configs/frozen.lock`. Run final dicatat di `results/final_runs.log`. |
| Leksikon | `resources/lexicon/seed_lexicon_v0.tsv` | **PLACEHOLDER 17 kata, tidak layak dilaporkan** |

Perintah utama (dijalankan dari folder `simplexid_experiment/`):

```
python experiments/run_h2.py  pairs | splits | templates | validate | auto-1a1o | status
python experiments/run_parse.py cache | stats | oracle | audit-sheet
python experiments/run_eval.py dev [--variants ...] | final (butuh freeze) | freeze --note "..." | gold-summary
python -m pytest -q
```

---

## 6. Hasil sejauh ini (semua dari split **dev**)

### 6.1 Oracle upper bound (H3.2, `reports/h32/oracle_dev.md`)

Analisis atas 1.593 pasangan gold dev: jarak antara head aspek dan head opini di pohon dependensi.

| Jarak maksimum | Recall maksimum |
|---|---|
| ≤1 | 73,5% |
| ≤2 | **94,0%** |
| ≤3 | 98,9% |

Rules saat ini hanya mencapai **51,0%** (head match). Artinya ada ruang perbaikan sekitar 43 poin dalam jarak 2.

Penyebab kegagalan terbesar:
- Kata opini diberi POS NOUN/NUM oleh parser ("kotor", "oke", "bau") dan terhubung lewat `compound`/`nummod` (~180 pasangan).
- `nsubj:pass` (37/40 gagal).
- `acl` (28/29 gagal).
- 23,8% head opini bukan ADJ/VERB; 11,3% head aspek bukan NOUN.
- Pasangan lewat jalur `conj`: 12,2%, dan hanya 30,8% yang tertangkap.

Pola jalur teratas:
- `NOUN vamod ADJ` 33,1% (sudah tertangkap semua);
- `NOUN vcompound NOUN` 7,4%;
- `NOUN ^nsubj VERB` 5,6%;
- `NOUN ^obj VERB` 4,8%.

### 6.2 Baseline harness (H3.3, `results/20261019-122748_dev_h33-baseline/`, commit 549de12, tree bersih)

| Varian | head P | head R | head F1 |
|---|---|---|---|
| V0 | 0,295 | 0,471 | 0,363 |
| V1 (+R3) | 0,284 | 0,485 | 0,358 |
| V2 (+R4) | 0,303 | 0,496 | 0,376 |
| V3 (+R3+R4) | 0,291 | 0,510 | 0,371 |
| V4–V9 | identik dengan V3 untuk ekstraksi | | |

- **Precision adalah masalah utama.** 707 FP di 519 kalimat yang gold-nya tanpa pasangan.
- Precision per rule (head): R2 **11%** (712 prediksi), R3 12%, R1 25%, R0 40%, R4 61%.
- R3 **menurunkan** F1, R4 menaikkan.
- Polaritas oracle-pair 0,282 untuk semua varian, karena leksikon placeholder. Efek DW dan SC **belum terukur**.
- R8: recall deteksi implisit 3,5%, akurasi kategori 3,5%. Baseline mayoritas (`general`) mencapai **75,5%**.

### 6.3 Evaluasi kandidat leksikon (hanya pada 400 pasangan DEV_POLARITY)

| Leksikon | Entri | Cakupan | Ketepatan tanda (dengan negasi) | Proxy akurasi | Lisensi |
|---|---|---|---|---|---|
| SentiStrengthID (masdevid) | 1.729 | 63,0% | **94,8%** | 60,4% | tidak ada file lisensi; turunan SentiStrength |
| ID-OpinionWords (masdevid, terjemahan Hu & Liu) | 3.442 | **71,5%** | 89,9% | **64,9%** | tidak ada file lisensi |
| InSet (Koto & Rahmaningtyas 2017) | 9.071 | 69,5% | 81,3% | 57,1% | tidak ada file lisensi ("Unknown") |
| SentiStrengthID (agusmakmun) | 1.548 | 54,2% | 94,5% | 51,8% | GPL-3.0 |
| Barasa (Indonesian SentiWordNet) | — | belum diuji | — | — | MIT + CC BY-SA 3.0 |

Catatan:
- InSet tidak cocok untuk domain hotel. 1.142 kata ada di daftar positif sekaligus negatif; "bagus" bernilai −2; "bersih" tidak ada.
- Kata yang tidak ada di semua leksikon: *strategis, berfungsi, disediakan, diperbaiki, tingkatkan*.

---

## 7. Keputusan yang sudah diambil (jangan diusulkan ulang tanpa bukti baru)

1. Split dev/final seperti §3, dengan final **terkunci**.
2. Metrik utama: **head match** F1. Strict dan overlap tetap dilaporkan.
3. **AUTO_1A1O hanya dipakai di dev, tidak pernah di final.** Buktinya: 228/228 kalimat 1A1O yang direview = A1-O1 (batas atas error ≈1,3% menurut *rule of three*). Metrik dilaporkan per strata dan per sumber. Versi gold di-hash, jadi verifikasi AUTO→REVIEWED otomatis mengubah versi.
4. Uji signifikansi: paired bootstrap per kalimat + McNemar + Holm, menggantikan K-fold + t-test.
5. Parser: UDPipe 2.5 GSD lewat `ufal.udpipe`, satu kalimat TermA = satu pohon.
6. Provenance: setiap run punya manifest; final run hanya setelah `freeze`; ringkasan run masuk git, file per-item tidak (kecuali run final).
7. Oracle split final **tidak dihitung** sampai H6.

## 8. Aturan integritas (tidak boleh dilanggar)

- Tidak ada keputusan desain, tuning, pemilihan leksikon/DAM, atau error analysis yang memakai TermA validation (final).
- `gold_pairs.csv` tidak diedit manual. Koreksi dilakukan di `gold_review_log.csv`, lalu jalankan `run_h2.py pairs`.
- Perintah `templates --overwrite` dan `auto-1a1o --overwrite` **menghapus anotasi**. Hindari.
- Perubahan pada gold final setelah `freeze` harus dicatat alasannya di `reports/research_log.md`.
- Angka dengan leksikon placeholder tidak boleh dilaporkan.
- Edit CSV di Excel berlocale Indonesia akan menyimpan pemisah `;` dan merusak validasi. Simpan sebagai CSV koma (UTF-8).

---

## 9. Keputusan tertunda (bantu pengguna memutuskan)

| # | Pertanyaan | Rekomendasi saat ini |
|---|---|---|
| D1 | Leksikon untuk H4 | SentiStrengthID (utama, berbobot) + ID-OpinionWords (cadangan). Ukur di dev. |
| D2 | Bangun leksikon domain hotel sendiri dari dev (kurasi ±300–500 kata)? | Ya. Bebas lisensi, sesuai domain, dan bisa jadi kontribusi artikel. Paralel: minta izin tertulis ke penulis SentiStrengthID. |
| D3 | Desain ulang ablation (§10) | Perlu persetujuan pengguna |
| D4 | Siapa annotator kedua untuk Cohen's κ? | Belum ada |
| D5 | Lisensi model parser non-komersial | Untuk produk komersial nanti perlu melatih parser dari UD_Indonesian-GSD (CC BY-SA). Diskusikan dengan HKI. |

---

## 10. Rancangan fase berikutnya (draft, minta AI menyempurnakan)

### H4: perbaikan sistem dan desain ulang ablation (dev saja)

1. **Leksikon** (D1/D2) + daftar negasi/booster terpadu. Saat ini daftar di `scoring.py` dan `gold.py` berbeda; `kurang/gak/nggak` belum ada di scoring.
2. **Precision:**
   - tinjau R2 (11%) dan R3 (12%);
   - kurangi FP pada kalimat tanpa aspek gold, misalnya filter aspek berbasis leksikon atau kamus aspek dari dev.
3. **Recall** (dari pola oracle):
   - kandidat opini berbasis leksikon untuk kata ber-POS NOUN/NUM;
   - `nsubj:pass` di R1;
   - `acl` menggantikan `relcl` di R7;
   - aspek PROPN/X ("AC" diberi POS X);
   - span opini mencakup modifier negasi/intensifier.
4. **Ablation baru:**
   - *Kumulatif*: Base → +R3 → +R4 → +R8 statis → +R8 kontekstual → +DW → +SC (= Full).
   - *Leave-one-out*: Full−R3, Full−R4, Full−R8ctx, Full−DW, Full−SC.
   - Implementasi **baseline pewarisan yang sungguhan** untuk SC: pasangan R3/R4 menyalin faktor dari pasangan asal.
5. **Pemetaan klaim → metrik → subset:**

| Mekanisme | Metrik | Subset |
|---|---|---|
| R3/R4 | pair F1 | COORD |
| R8 | akurasi / macro-F1 kategori vs baseline mayoritas | IMPLICIT, IMPLICIT_AMBIG |
| DW | oracle-pair polarity accuracy / macro-F1 | NEGATION, INTENSIFIER |
| SC | oracle-pair polarity | **ASYM_MOD** |

6. **R8 perlu dirancang ulang.** 77% gold implisit adalah `general` (pujian umum). DAM sekarang hanya berisi 13 kata, berfokus ac/wifi. Disambiguasi kontekstual tidak punya konteks di level kalimat. Pertimbangkan:
   - DAM dibangun dari data dev (HoASA + TermA train);
   - evaluasi IMPLICIT_AMBIG dengan konteks multi-kalimat bila memungkinkan;
   - melaporkan jujur jika R8 kalah dari baseline mayoritas.
7. **Tuning** `NEG_DISTANCE_THRESHOLD` dan `INT_DECAY_LAMBDA` lewat grid search di dev. Laporkan sensitivitasnya.
8. **Baseline eksternal:** proximity sederhana, Double Propagation, dan opsional IndoBERT sebagai upper bound neural.

### H5: validasi manusia (lihat §11)

### H6: pre-registration → `freeze` → satu kali run final

### H7: tabel efek teknis, error analysis, draf artikel (setelah konsultasi HKI)

---

## 11. Panduan review gold (H5)

### 11.1 Prioritas

| Urutan | File | Baris yang dicek dulu | Menghambat |
|---|---|---|---|
| 1 | `gold_dev_auto_1a1o.csv` | Semua 468 (sedang dikerjakan pengguna) | Finalisasi tuning H4 |
| 2 | `gold_implicit.csv` | **356 MEDIUM/LOW** (194+1 dev, 160+1 final) | H6 |
| 3 | `gold_polarity.csv` | **31 MEDIUM** (14 dev, 17 final) + sampel acak HIGH | H6 |
| 4 | `gold_review_log.csv` | Sampel acak ±10% + ±20% oleh annotator kedua (κ) | H6 |
| 5 | `gold_pairs.csv` kolom `secondary_tags` | Tag COORD / IMPLICIT_AMBIG / ASYM_MOD (auto-tag lalu verifikasi) | H6 |
| 6 | `data/annotation/parser_audit/parser_audit_sheet.csv` | 100 kalimat / 1.144 token (panduan di README folder itu) | Artikel (*limitations*) |

### 11.2 Aturan per file

**A. `gold_dev_auto_1a1o.csv`.** Apakah satu-satunya aspek (A1) benar-benar dinilai oleh satu-satunya opini (O1)?

| Kondisi | Isi |
|---|---|
| Ya, berpasangan | `annotation_status=REVIEWED` |
| Tidak berhubungan (opini menilai hal lain / hotel secara umum) | `accepted_pairs=NONE`, `REVIEWED` |
| Ragu | `accepted_pairs=AMBIGUOUS`, `REVIEWED`, alasan di `notes` |

Validasi: `python experiments/run_h2.py auto-1a1o`.

**B. `gold_review_log.csv`.** Untuk setiap A<i> dan O<j>, apakah O<j> menilai A<i>?
- Satu opini boleh menilai beberapa aspek ("kamar dan kamar mandi bersih" → A1-O1|A2-O1), dan sebaliknya.
- Opini yang menilai hotel secara umum, atau aspek yang tidak ada di daftar, **tidak** dipasangkan.
- Jika tidak ada pasangan sama sekali, isi `NONE`. Jika tidak bisa diputuskan, isi `AMBIGUOUS`.
- Span tidak boleh diubah (span berasal dari TermA). Catat jika span TermA tampak salah, tetapi tetap jangan diubah.
- Setelah koreksi, jalankan `run_h2.py pairs` lalu `run_h2.py validate`.

**C. `gold_polarity.csv`.** Polaritas pasangan **dalam konteks kalimat**, dengan negasi/intensifier di dalam span opini ikut dihitung.
- POSITIVE / NEGATIVE.
- NEUTRAL hanya jika benar-benar tidak ada orientasi evaluatif (pernyataan faktual atau netral). Saat ini hanya ada 11 baris NEUTRAL.
- Kasus sulit yang sering muncul:
  - "**tidak ada**": "tidak ada kendala" → POSITIVE, "air panas tidak ada" → NEGATIVE;
  - "kurang dingin" → NEGATIVE;
  - "lumayan bagus" → POSITIVE;
  - sarkasme: nilai sesuai maksud.
- Catat confidence baru di `notes` jika keputusannya diubah.

**D. `gold_implicit.csv`.** Opini tanpa aspek eksplisit di kalimat.
- `decision=IMPLICIT`: opini menilai suatu aspek hotel yang tidak disebut. Isi `gold_aspect_category` dengan salah satu kategori berikut:

| Kategori | Cakupan |
|---|---|
| `ac` | pendingin ruangan / suhu kamar |
| `air_panas` | air panas / shower |
| `bau` | aroma / bau |
| `general` | hotel atau pengalaman menginap secara umum ("mantap", "puas", "recommended") |
| `kebersihan` | kebersihan secara umum |
| `linen` | seprai, handuk, bantal, selimut |
| `service` | staf / pelayanan / check-in |
| `sunrise_meal` | sarapan |
| `tv` | televisi |
| `wifi` | internet |

- `decision=NOT_IMPLICIT`: bukan penilaian aspek hotel (salam, terima kasih, harapan, cerita).
- `decision=AMBIGUOUS`: tidak bisa diputuskan.
- Isi juga `gold_polarity`.
- Waspadai bias ke `general`. Pilih kategori spesifik bila kalimat memberi petunjuk, misalnya "dingin banget" → `ac`, "wangi" → `bau`.
- Validasi: `python experiments/run_h2.py validate`.

### 11.3 Prosedur yang disarankan

1. Verifikasi buta dulu (tanpa melihat label lama) pada sampel, lalu bandingkan dengan label lama. Ini mengukur keandalan label berbantuan AI.
2. **Annotator kedua** mengerjakan sampel yang sama secara independen (±20% per file, stratified) → hitung **Cohen's κ**. Target κ ≥ 0,70 (pasangan & polaritas). Untuk kategori implisit, laporkan κ apa adanya.
3. Ketidaksepakatan diselesaikan lewat adjudikasi; alasannya dicatat di `notes`.
4. Semua perubahan pada gold **final** dicatat di `reports/research_log.md` (tanggal, jumlah baris, alasan).
5. Setelah selesai: jalankan `pytest`, `run_h2.py validate`, lalu `run_eval.py gold-summary`. Versi gold baru dicatat.

**Bantuan yang diharapkan dari AI pada H5:**
- Menjelaskan dan menyempurnakan pedoman anotasi di atas menjadi *annotation guideline* formal untuk artikel.
- Merancang sampling untuk κ: ukuran sampel, stratifikasi, dan cara menghitung.
- Membantu memutuskan kasus sulit yang ditempelkan pengguna, dengan alasan singkat.
- **Tidak** melabeli massal tanpa diperiksa pengguna.

---

## 12. Glosarium singkat

| Istilah | Arti |
|---|---|
| head match | prediksi benar jika head token aspek dan opini prediksi berada di dalam span gold |
| oracle-pair | pasangan gold dimasukkan langsung ke modul skor, untuk memisahkan efek formula dari error ekstraksi |
| 1A1O | kalimat dengan tepat 1 span aspek dan 1 span opini |
| strata | EMPTY (gold tanpa pasangan) / SIMPLE (1A1O) / COMPLEX |
| COORD / IMPLICIT_AMBIG / ASYM_MOD | subset uji untuk R3/R4, R8 kontekstual, dan scope consistency. ASYM_MOD: konjungsi di mana negasi/intensifier hanya menempel ke salah satu konjungtor, misalnya "sangat lemah dan tidak stabil". |
| freeze | mengunci hash kode/resource/gold sebelum final run |

## 13. Referensi utama

- Qiu, Liu, Bu, Chen (2011). *Opinion word expansion and target extraction through double propagation.* Computational Linguistics.
- Taboada et al. (2011). *Lexicon-based methods for sentiment analysis* (SO-CAL). Computational Linguistics.
- Hu & Liu (2004); Liu, Hu, Cheng (2005). *Opinion Observer.* WWW.
- Wilie et al. (2020). *IndoNLU* (TermA, HoASA).
- Koto & Rahmaningtyas (2017). *InSet Lexicon.* IALP.
- Wahid & Azhari (2016). IJCCS 10(2), basis SentiStrengthID / ID-OpinionWords.
- Straka & Straková (2019). *UD 2.5 Models for UDPipe.* LINDAT, hdl 11234/1-3131.
