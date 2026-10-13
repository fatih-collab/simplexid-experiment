# UDPipe model — provenance

| Field | Value |
|---|---|
| File | `indonesian-gsd-ud-2.5-191206.udpipe` |
| Collection | Universal Dependencies 2.5 Models for UDPipe (2019-12-06), Straka & Straková |
| Handle | http://hdl.handle.net/11234/1-3131 |
| Download URL | https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/d6a628a3-bb7b-4ea3-a608-f1b2f08407f7/content |
| Size | 14,257,882 bytes |
| MD5 (published by LINDAT) | `e0167ecc9d9a953bc14726b6e928725e` |
| SHA-256 | `a91987cf8dba83f730d9828937016c70c3539728dc0de6608ff089b075fc3e1d` |
| License | CC BY-NC-SA 4.0 (**non-commercial**) |
| Downloaded | 2026-10-13 |

The model file itself is not committed to git (see `.gitignore`); re-download it from the URL above and verify the checksum:

```bash
curl -L -o indonesian-gsd-ud-2.5-191206.udpipe "<Download URL>"
md5sum indonesian-gsd-ud-2.5-191206.udpipe   # must equal the MD5 above
```

Why not `spacy_udpipe.download("id")`: LINDAT migrated to DSpace 7 and the legacy
`/xmlui/bitstream/...` URL used by spacy-udpipe 1.0.0 now returns an HTML page
(~465 KB), which spacy-udpipe silently saves as the model file.

License note: CC BY-NC-SA forbids commercial use of this model. It is fine for research
and for the paper, but any commercial deployment of SimplEx-ID needs a parser trained
on a treebank whose license allows it (UD_Indonesian-GSD itself is CC BY-SA 4.0).
