<div align="center">

# pdffff

**Durable, fast, full-folder search across PDFs — TUI and desktop GUI.**

Point pdffff at a folder, it indexes, watches, and searches as you type —
literal, regex, or fuzzy — in milliseconds.

[Website](https://lambdabetaeta.github.io/pdffff/) ·
[Architecture](docs/architecture.md)

</div>

---

## At a glance

- **`pdftotext`** (Poppler) extracts text from each PDF.
- **SQLite** (WAL, `synchronous=NORMAL`) is the durable, on-disk store.
- An in-process **dense bigram inverted index** turns each query into a
  candidate set in microseconds; candidates are verified exactly with
  `memchr::memmem` (literal), `regex` (regex), or `neo_frizbee` (fuzzy).
- A small **mutation overlay** (tombstones + overflow chunks) lets a
  `notify`-debounced watcher reflect file changes within ~200 ms without
  rebuilding the base. The base is atomically rebuilt out of band when
  the overlay crosses configurable thresholds.

No daemon. No always-on service. One binary that opens a SQLite file,
loads its index into memory, and answers queries.

## Install

**Runtime dependency** — Poppler's `pdftotext`:

| Platform        | Command                          |
|-----------------|----------------------------------|
| Debian/Ubuntu   | `sudo apt-get install poppler-utils` |
| Fedora          | `sudo dnf install poppler-utils`     |
| macOS           | `brew install poppler`               |

**Build from source** (Rust 1.74+):

```sh
git clone https://github.com/lambdabetaeta/pdffff
cd pdffff
cargo install --path .              # builds the TUI binary `pdffff`
cargo install --path . --features gui --bin pdffff-gui
```

The GUI is optional — gated behind the `gui` Cargo feature so the
default build doesn't pull in `eframe`/`winit`. Without `cargo install`,
`cargo run --release -- …` works equally well.

## Usage

```sh
pdffff ~/papers           # interactive TUI
pdffff-gui ~/papers       # Win9x-themed desktop GUI (--features gui)
```

The initial scan and per-PDF extraction run in the background — results
appear as soon as the first chunks land in the index, and the status
bar shows live progress.

| Key                                   | Action                              |
|---------------------------------------|-------------------------------------|
| any character                         | append to the query                 |
| `Backspace` / `Ctrl+W` / `Ctrl+U`     | erase / word-erase / clear query    |
| `↑` `↓` / `Ctrl+P` `Ctrl+N`           | move selection                      |
| `PageUp` / `PageDown`                 | page through results                |
| `Tab`                                 | cycle mode: literal → regex → fuzzy |
| `Enter`                               | open the selected hit in your PDF viewer |
| `Esc` / `Ctrl+C` / `Ctrl+D` / `Ctrl+Q` | quit                               |

Enter on a result opens the file in the host's default viewer (`open` on
macOS, `xdg-open` on Linux, `cmd /C start` on Windows) and **keeps the
session running** so you can pick more results back-to-back.

### Selector mode

Pass `--print-path` and Enter prints the chosen path to stdout instead
of opening it, so pdffff composes with shell pipelines:

```sh
cd "$(dirname "$(pdffff ~/papers --print-path)")"
pdffff ~/papers --print-path | xargs -I{} mv {} ~/inbox/
```

### Paths

The SQLite database lives per-corpus under your platform's data
directory — `$XDG_DATA_HOME/pdffff/<basename>-<hash>.db` on Linux,
the macOS / Windows equivalents elsewhere. Override with
`--db /path/to/file.db`. Tracing output goes to `$TMPDIR/pdffff.log` by
default; override with `--log-file`.

## How it works

`pdftotext - -` is invoked per PDF (PDF on stdin, text on stdout, the
same shape `rga`'s built-in adapter uses). The output is split on
`\x0c`, normalized through `deunicode` + lowercase + whitespace-collapse,
and sliced into 1200-character chunks with 200-character overlap. Chunks
are persisted in SQLite.

On startup the chunk table is streamed into a flat `Vec<ChunkItem>` plus
a 65 536-column dense bigram posting-list index (adapted from `fff`'s
`bigram_filter.rs`). Queries first ask the index for a candidate bitset
of chunks that might contain the query —

- **literal**: AND of consecutive bigrams + skip-1 bigrams;
- **regex**: AND/OR tree derived from `regex-syntax`;
- **fuzzy**: OR-of-AND of evenly-spaced probe bigrams;

— then verify each candidate exactly. A small `Overlay` (tombstones +
overflow chunks) layered over the dense base index lets the
`notify-debouncer-full` watcher reflect new / modified / deleted PDFs
within ~200 ms. `arc-swap` atomically rebuilds the base out of band when
the overlay crosses thresholds, with zero interruption to live queries.

See [`docs/architecture.md`](docs/architecture.md) for the
developer-facing layout.

## Status

All five measurable success criteria from the design report hold:

1. Indexes many PDFs into SQLite without manual intervention.
2. Second startup re-uses SQLite (zero re-extraction).
3. Literal query over 5 000 chunks: p95 latency well under 50 ms.
4. Modified PDF surfaces in search within < 2 s via the watcher.
5. Every result carries a path, a 1-based page number, and a
   match-centred snippet.

`cargo test` is green; `cargo build --release` and
`cargo build --release --features gui` are both clean.

## Attribution

- The dense bigram index (`src/bigram.rs`) and the regex/fuzzy
  bigram-query decomposition (`src/bigram_query.rs`) are adapted from
  the [`fff`](https://github.com/dmtrKovalenko/fff) project,
  © 2025 Dmitriy Kovalenko, MIT.
- The `pdftotext - -` extraction shape and SQLite-as-extracted-text
  store decision are taken from
  [`rga`](https://github.com/phiresky/ripgrep-all) (Apache-2.0).
- `pdftotext` itself comes from
  [Poppler](https://poppler.freedesktop.org/) (GPL-2.0-or-later).
  pdffff invokes it as a subprocess; it is not linked.

## License

Released under the [MIT License](LICENSE). All transitive dependencies
are MIT-compatible (mostly MIT-or-Apache-2.0 dual-licensed, with a few
BSD/ISC/Zlib/CC0/Unlicense for good measure).
