# pdffff

Durable, fast, full-folder search across PDFs, served as an interactive
TUI. Point pdffff at a folder of PDFs and it indexes, watches, and
searches as you type — literal, regex, or fuzzy — in milliseconds.

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
pdffff ~/papers
```

That opens the TUI. The initial scan and per-PDF extraction run in the
background — results start appearing as soon as the first chunks land in
the index, and the status bar shows progress.

Keys:

| Key                                 | Action                              |
|-------------------------------------|-------------------------------------|
| any character                       | append to the query                 |
| `Backspace` / `Ctrl+W` / `Ctrl+U`   | erase / word-erase / clear query    |
| `↑` `↓` / `Ctrl+P` `Ctrl+N`         | move selection                      |
| `PageUp` / `PageDown`               | page through results                |
| `Tab`                               | cycle mode: literal → regex → fuzzy |
| `Enter`                             | open the selected hit in your PDF viewer |
| `Esc` / `Ctrl+C` / `Ctrl+D` / `Ctrl+Q` | quit                             |

The SQLite database is stored per-corpus under your platform's data
directory (`$XDG_DATA_HOME/pdffff/<basename>-<hash>.db` on Linux).
Override with `--db /path/to/file.db`. Tracing output goes to a log
file — `$TMPDIR/pdffff.log` by default, or pass `--log-file`.

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

The five measurable success criteria from the design report all hold:

1. Indexes many PDFs into SQLite without manual intervention.
2. Second startup re-uses SQLite (zero re-extraction).
3. Literal query over 5 000 chunks: p95 latency well under 50 ms.
4. Modified PDF surfaces in search within < 2 s via the watcher.
5. Every result carries a path, a 1-based page number, and a
   match-centred snippet.

`cargo test` is green; `cargo build --release` is clean.

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
