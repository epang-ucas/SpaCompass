# Runtime resources

These files are part of the release: the tokeniser needs them to map gene ids to
the model vocabulary, and platform-mean normalisation uses the mean-expression
tables.

| File | Contents | Used by | SHA-256 |
| --- | --- | --- | --- |
| `vocab/human_merged_gene_to_ensembl.json` | human gene symbol → Ensembl id | gene tokenisation (`--species human`, symbols) | `ce2ad9b9bf6d9a206652f153d4f7a0c4e973a409dee1f3a188e1fe1b920f6b10` |
| `vocab/human_vocab.json` | human Ensembl id → model token id | gene tokenisation | `c07eb340e241a39fc7629dcd9edfd0996efaf3adb1f130da1da1f4eb6ac01e11` |
| `vocab/mouse_merged_gene_to_ensembl.json` | mouse gene symbol → Ensembl id | gene tokenisation (`--species mouse`, symbols) | `be36efdde79bfc1d71a8fcd7957f87a45f77ca22606e27e913923dc639b33e70` |
| `vocab/mouse_vocab.json` | mouse Ensembl id → model token id | gene tokenisation | `dd7ec1b99a257f8413596c79974407bb855b1b42df81004fd13aa55cd0fdef5c` |
| `platform_mean/human_platform_mean.tsv` | mean expression per gene and platform | optional platform-mean normalisation (`--platform-mean-file`) | `be5fcb3dffa7b0e9490b56b8cf5efad5c6a1fe26389a9fdce263e60e73325b6c` |
| `platform_mean/mouse_platform_mean.tsv` | mean expression per gene and platform | optional platform-mean normalisation | `5c64bec94d78408438c4f592b4593bc870b1e411b46a29f8ca3c371d33c2f2e7` |

The scripts locate them automatically relative to the repository root. Set
`SPACOMPASS_RESOURCES` to load them from another directory, for example when the
files live on a shared volume:

```bash
export SPACOMPASS_RESOURCES=/path/to/resources
```
