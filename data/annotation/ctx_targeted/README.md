# Anotasi tertarget disambiguasi kontekstual R8 (dev)

Isi `ctx_targeted_dev.csv`. **Jangan membuka `ctx_targeted_dev_system.csv` sebelum selesai**: penilaian harus buta.

Setiap baris menanyakan: kata `opinion_text` dalam `sentence` menilai aspek hotel yang mana?

| Kolom | Isi | Arti |
|---|---|---|
| `decision` | `IMPLICIT` | Kata itu menilai aspek hotel yang tidak disebut sebagai pasangannya di kalimat |
| | `NOT_IMPLICIT` | Bukan penilaian aspek hotel, atau jelas menilai aspek yang tertulis tepat di sekitarnya |
| | `AMBIGUOUS` | Tidak bisa diputuskan |
| `gold_aspect_category` | salah satu kategori | Hanya jika `IMPLICIT` |
| `annotation_status` | `REVIEWED` | Tandai setelah baris selesai |
| `notes` | bebas | Opsional |

Kategori: `ac`, `air_panas`, `bau`, `general` (hotel/pengalaman secara umum), `kebersihan`, `linen` (seprai, handuk, bantal, selimut), `service`, `sunrise_meal` (sarapan), `tv`, `wifi`.

Pilih kategori spesifik bila kalimat memberi petunjuk; `general` jika memang menilai secara umum.
Baris boleh dicicil. Setelah selesai, simpan ke path yang sama, lalu jalankan:

```
python experiments/run_ctx_targeted.py evaluate
```
