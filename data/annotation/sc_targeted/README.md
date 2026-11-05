# Anotasi tertarget scope consistency (dev)

Isi `sc_targeted_dev.csv`. **Jangan membuka `sc_targeted_dev_system.csv` sebelum selesai**: file itu berisi
prediksi sistem, dan penilaian harus buta agar tidak bias.

Setiap baris adalah satu pasangan aspek–opini yang dihasilkan sistem. Baca `sentence`, lalu isi:

| Kolom | Isi | Pertanyaan |
|---|---|---|
| `is_valid_pair` | `YES` / `NO` / `AMBIGUOUS` | Apakah `opinion_text` benar-benar menilai `aspect_text` dalam kalimat ini? |
| `gold_polarity` | `POSITIVE` / `NEGATIVE` / `NEUTRAL` | Hanya jika `YES`: sentimen opini terhadap aspek ini, menurut konteks kalimat |
| `annotation_status` | `REVIEWED` | Tandai setelah baris selesai |
| `notes` | bebas | Opsional; tulis alasan jika ragu |

Pedoman:

- Nilai sesuai **konteks kalimat**, termasuk negasi di sekitar opini, walaupun tidak tertulis di `opinion_text`.
  Contoh: pada "sangat lemah dan tidak stabil", pasangan (sinyal, stabil) bernilai `NEGATIVE`.
- `NO` jika opini menilai hal lain. Contoh: pada "kamar bersih dan ac dingin", (kamar, dingin) adalah `NO`.
- `NEUTRAL` hanya jika benar-benar tidak ada penilaian.
- Baris boleh dicicil; yang belum `REVIEWED` diabaikan.

Setelah selesai, simpan ke path yang sama (Excel dengan pemisah `;` tetap terbaca), lalu jalankan:

```
python experiments/run_sc_targeted.py evaluate
```

Membuat ulang dengan `build --overwrite` **menghapus anotasi**.
