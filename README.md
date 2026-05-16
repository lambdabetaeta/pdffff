# pdffff

Durable, fast, full-folder search across PDFs. Run a one-shot `index` over a
directory of PDFs, then `search` for a phrase, a regex, or a fuzzy approximation
of one — in milliseconds. Or run `watch` and let pdffff keep itself up to date
as files come and go.

The architecture follows the design laid out in
[`deep-research-report.md`](deep-research-report.md):

* **`pdftotext`** (Poppler) extracts text from each PDF.
* **SQLite** (WAL) is the durable store for `documents` and `chunks`.
* An in-process **dense bigram inverted index** turns each query into a
  candidate set of chunks in microseconds; the candidates are then verified
  with `memchr::memmem` (literal), the `regex` crate (regex), or
  `neo_frizbee` (fuzzy).
* A small **mutation overlay** (tombstones + overflow chunks) lets a
  `notify`-based watcher make file changes visible within ~200 ms without
  rebuilding the base index. The base is atomically rebuilt out of band
  when the overlay crosses configurable thresholds.

No daemon. No always-on service. One self-contained binary that opens a
SQLite file, loads the index into memory, and answers queries.

## Install

```sh
# Runtime dependency:
#   Debian/Ubuntu:  sudo apt-get install poppler-utils
#   Fedora:         sudo dnf install poppler-utils
#   macOS:          brew install poppler

git clone https://github.com/user/pdffff
cd pdffff
cargo install --path .
```

The binary is `pdffff`. Without `cargo install`, `cargo run --release -- …`
works just as well.

## Usage

```sh
# Index a folder of PDFs into ./pdffff.db
pdffff index ~/papers

# Look something up
pdffff search "monad transformer"            # literal
pdffff search --mode regex 'fn\s+main'        # regex
pdffff search --mode fuzzy 'topolgy'          # fuzzy / typo-tolerant
pdffff search --json "deep learning" | jq .   # JSON output, one hit per line

# Live mode — watches the folder and answers queries from stdin
pdffff watch ~/papers

# Diagnostics
pdffff info       # row counts + bigram heap size
pdffff diagnose   # pdftotext version, SQLite integrity_check, errored docs
```

By default the SQLite database lives in `./pdffff.db`. Override with
`--db /path/to/file.db`.

## Output format

Human-readable (default):

```text
1. /home/me/papers/perceptron.pdf (page 3, chunk #7, score 1.00)
     a perceptron is the simplest kind of artificial neural network …

2. /home/me/papers/backprop.pdf (page 1, chunk #0, score 1.00)
     introduction to backpropagation and the perceptron learning rule …
```

With a TTY, the filename is bold, the metadata is dim, and the matched
phrase / terms are rendered with inverse video. `NO_COLOR=1` disables
colors.

JSON (`--json`):

```jsonl
{"chunk_id":12,"doc_id":3,"path":"/home/me/papers/perceptron.pdf","page_no":3,"chunk_ord":7,"score":1.0,"snippet":"a perceptron is the simplest kind of artificial neural network …"}
```

## Subcommands

| Command            | Purpose                                                            |
|--------------------|--------------------------------------------------------------------|
| `pdffff scan ROOT` | Dry-run scanner; shows which PDFs would be (re-)extracted.         |
| `pdffff index ROOT`| One-shot index/refresh.                                            |
| `pdffff watch ROOT`| Live mode: scan + extract, then watch, then interactive REPL.      |
| `pdffff search Q`  | Search the indexed corpus.                                         |
| `pdffff rebuild`   | Force an in-memory base-index rebuild from SQLite. Diagnostics.    |
| `pdffff info`      | Document counts by status, active chunks, bigram heap MiB.         |
| `pdffff diagnose`  | `pdftotext` version, SQLite integrity check, errored documents.    |

`pdffff <cmd> --help` for the full flag list of each subcommand.

## How it works (one paragraph)

`pdftotext - -` is invoked per PDF (PDF on stdin, text on stdout, same as
`rga`'s built-in adapter). The output is split on `\x0c`, normalized
through `deunicode` + lowercase + whitespace-collapse, and sliced into
1200-character chunks with 200-character overlap. Chunks are persisted in
SQLite. On startup the chunk table is streamed into a flat
`Vec<ChunkItem>` plus a 65536-column dense bigram posting-list index
(adapted from `fff`'s `bigram_filter.rs`, MIT). Queries first ask the
index for a candidate bitset of chunks that might contain the query
(literal: AND of consecutive bigrams + skip-1 bigrams; regex: AND/OR tree
from `regex-syntax`; fuzzy: OR-of-AND of evenly spaced probe bigrams),
then verify each candidate exactly. A small `Overlay` (tombstones +
overflow chunks) layered over the dense base index lets the
`notify-debouncer-full`-based watcher reflect new/modified/deleted PDFs
within ~200 ms; the base is atomically rebuilt by `arc-swap` when the
overlay crosses thresholds.

## Status

All seven days of the deep-research roadmap are implemented and the
five measurable success criteria are exercised in `tests/success_criteria.rs`:

1. Indexes many PDFs into SQLite without manual intervention.
2. Second startup re-uses SQLite (zero re-extraction).
3. Literal query over 5 000 chunks: p95 latency well under 50 ms.
4. Modified PDF surfaces in search within < 2 s via the watcher.
5. Every result carries a path, a 1-based page number, and a
   match-centred snippet.

`cargo test` runs 97 tests (70 unit + 27 integration); `cargo build
--release` is clean.

## Attribution

* The dense bigram index (`src/bigram.rs`) and the regex/fuzzy bigram-query
  decomposition (`src/bigram_query.rs`) are adapted from the
  [`fff`](https://github.com/dmtrKovalenko/fff) project, © 2025 Dmitriy
  Kovalenko, MIT.
* The `pdftotext - -` extraction shape and SQLite-as-extracted-text-store
  decision are taken from [`rga`](https://github.com/phiresky/ripgrep-all)
  (Apache-2.0).
* `pdftotext` itself comes from [Poppler](https://poppler.freedesktop.org/),
  GPL-2.0+. pdffff invokes it as a subprocess; it is not linked.

## License

Dual-licensed under MIT and Apache-2.0. See
[`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE).
