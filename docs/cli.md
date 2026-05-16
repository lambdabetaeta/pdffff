# pdffff CLI reference

Every subcommand is a thin wrapper around a function in
[`src/app.rs`](../src/app.rs). The library never writes to stdout; the
binary in [`src/main.rs`](../src/main.rs) owns all formatting.

## Global flags

```
--db PATH    SQLite database file (default: ./pdffff.db)
```

`RUST_LOG=info` (or `debug`, `trace`) controls structured logging via
`tracing-subscriber`.

`NO_COLOR=1` forces plain-text output even on a TTY.

## `pdffff scan ROOT`

Dry-run: walk `ROOT` and report what would be indexed without running
`pdftotext`. Useful to verify `.gitignore` and symlink interactions before
committing to a long extraction run.

```
$ pdffff scan ~/papers --respect-ignore
scanned 312 files; 11 need extraction, 0 disappeared
  [New] /home/me/papers/2026-04-attention-is-all-you-need-redux.pdf
  [Modified] /home/me/papers/perceptron.pdf
  [StaleExtractor] /home/me/papers/topology-notes.pdf
  ...
```

`DirtyReason` values:

* `New` — never seen before.
* `Modified` — `mtime_ns` or `size_bytes` changed.
* `StaleExtractor` — `extractor` / `extractor_version` / `norm_version`
  mismatch.
* `RetryAfterError` — was `status='error'` last time; try again.

## `pdffff index ROOT`

One-shot scan + extract + UPSERT. Uses a `rayon` pool of `min(num_cpus, 6)`
extractors and a single DB-writer thread.

```
$ pdffff index ~/papers
indexed: seen=312 dirty=11 ok=10 empty=1 error=0 deleted=2 elapsed=4.81s
```

Flags:

```
--respect-ignore       Respect .gitignore / .ignore files.
--follow-symlinks      Follow symlinks during the walk.
--jobs N               Override extractor pool size (default min(num_cpus, 6)).
```

## `pdffff watch ROOT`

Long-running: one synchronous scan + extract pass, then a debounced
`notify` watcher that keeps the in-memory index live. After the initial
sync the process prints `> ` and reads literal queries from stdin until
EOF or an empty line.

```
$ pdffff watch ~/papers
watching /home/me/papers (debounce 200 ms). type a literal query and press enter; empty line or EOF to quit.
> hyperbolic
1. /home/me/papers/space-forms.pdf (page 4, chunk #2, score 1.00)
     hyperbolic space H^n is the simply-connected complete riemannian …
>
```

Flags:

```
--respect-ignore       Respect .gitignore / .ignore files.
--follow-symlinks      Follow symlinks during the walk.
--jobs N               Override extractor pool size (default min(num_cpus, 6)).
--debounce-ms N        Watcher debounce window (default 200, range 50–250).
```

## `pdffff search QUERY`

Searches the SQLite-backed corpus. The base index is loaded from SQLite
fresh on every invocation — the cost is dominated by the initial
`SELECT … FROM chunks` and the dense bigram build, both of which are
linear in chunk count.

```
$ pdffff search 'monad transformer'
1. /home/me/papers/mtl.pdf (page 1, chunk #0, score 1.00)
     a monad transformer is a higher-order construct that takes a monad …
```

Flags:

```
--mode MODE      literal (default) | regex | fuzzy
--limit N        Cap printed hits (default 200, the report's DISPLAY_LIMIT)
--json           One compact JSON object per hit, one per line.
```

Mode notes:

* `literal` — exact substring match on `text_norm_ascii` (the normalized
  pipeline). Case-insensitive because both index and query are lowercased
  through `deunicode + lowercase + whitespace-collapse`.
* `regex` — regex compiled via `regex::RegexBuilder::new(p).case_insensitive(true)`.
  Matched against `text_utf8` (the original, non-normalized text) so
  patterns like `\d{4}-\d{2}-\d{2}` work as expected.
* `fuzzy` — `neo_frizbee` over a rank text of
  `"{filename} {path} page {page_no} {preview}"`. The query is
  normalized; up to `min(query_len/3, 2)` typos are tolerated. The bigram
  prefilter retains chunks that contain `n - max_typos` of `n=6` evenly
  spaced probe bigrams.

## `pdffff rebuild`

Force an in-memory base-index rebuild from SQLite (without re-extracting
PDFs). Useful for diagnostics and to validate that the durable corpus is
round-trip-clean. The long-running `watch` mode does this automatically
when the overlay crosses thresholds.

```
$ pdffff rebuild
rebuild: docs=312 chunks=2840 bigram_bytes=18874368 elapsed=0.21s
```

## `pdffff info`

One-shot report:

```
$ pdffff info
documents: 312 total
  ok:      310
  empty:   1
  error:   1
  deleted: 0
chunks:    2840 active
bigram heap: 18874368 bytes (18.00 MiB)
```

## `pdffff diagnose`

End-to-end install + DB + corpus check. Runs:

1. `pdftotext -v` and prints the version banner.
2. `PRAGMA integrity_check` against the SQLite file.
3. Document-count breakdown by status.
4. Up to 20 most recent `status='error'` rows with their stored
   `error_text`.

```
$ pdffff diagnose
== pdftotext ==
  ok: pdftotext version 22.12.0

== sqlite ==
  integrity_check: ok
  documents: 312 total
    ok:      310
    empty:   1
    error:   1
    deleted: 0

== errored documents (up to 20) ==
  /home/me/papers/scanned-old.pdf :: pdftotext exited nonzero: Internal Error: PDF file is damaged
```
