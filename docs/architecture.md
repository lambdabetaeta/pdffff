# pdffff architecture

This is the developer-facing companion to
[`deep-research-report.md`](../deep-research-report.md). The report
specifies *what* to build and *why*; this file says *where* each piece
lives in the source tree and what its surface is.

## The boundary

```
┌──────────────────────────── persistence ───────────────────────────┐
│  SQLite (WAL)                                                       │
│    documents — re-extraction contract: path, size, mtime_ns,        │
│                extractor + extractor_version + norm_version,        │
│                status (ok | empty | error | deleted).               │
│    chunks    — text_utf8 (display), text_norm_ascii (search-norm),  │
│                page_no, chunk_ord, char_start/end, preview.         │
└─────────────────────────────────────────────────────────────────────┘
                            ▲                    ▲
                  startup load                    │
                            │                    │ writer-thread UPSERTs
┌────────────────── in-memory hot path ──────────┴──────────────────┐
│  BaseIndex                                                          │
│    Arc<Vec<ChunkItem>>                                              │
│    doc_ranges: HashMap<doc_id, Range<usize>>                        │
│    Arc<BigramIndex>                                                  │
│                                                                     │
│  Overlay  (parking_lot::RwLock)                                     │
│    tombstones: Vec<u64>           // 1 bit per base chunk           │
│    overflow_chunks: Vec<ChunkItem>                                  │
│    overflow_bigrams: Vec<Vec<u16>>                                  │
│    changed_docs: HashSet<doc_id>                                    │
│                                                                     │
│  IndexState                                                         │
│    base: ArcSwap<BaseIndex>                                         │
│    overlay: RwLock<Overlay>                                         │
└─────────────────────────────────────────────────────────────────────┘
                            ▲                    ▲
                  candidates +                  │
                  verification               mutation
                            │                    │
                       query::search        run_watch coordinator
```

`BigramIndex` is candidate-generation only; the query engine never decides
hits from a posting list intersection. The verification stage (`memchr`,
`regex::Regex`, `neo_frizbee`) is the source of truth for whether a chunk
is really a match.

## Modules

| File                    | Owns                                                                |
|-------------------------|---------------------------------------------------------------------|
| `src/db.rs`             | SQLite schema, migrations, UPSERT/load helpers, statuses.            |
| `src/normalize.rs`      | `deunicode` + lowercase + whitespace-collapse. `NORM_VERSION`.       |
| `src/extract.rs`        | `pdftotext - -` subprocess, page split on `\x0c`, chunker (1200/200). |
| `src/scanner.rs`        | `ignore::WalkBuilder` traversal + diff against `documents`.          |
| `src/watcher.rs`        | `notify-debouncer-full` → flume `WatchEvent`.                        |
| `src/bigram.rs`         | Dense `BigramIndex` (adapted from `fff`, MIT).                       |
| `src/bigram_query.rs`   | `regex_to_bigram_query`, `fuzzy_to_bigram_query` (adapted from `fff`).|
| `src/index.rs`          | `ChunkItem`, `BaseIndex`, `Overlay`, `IndexState`, rebuild routine.   |
| `src/query.rs`          | `search(state, query, mode, limit)`, snippet rendering.              |
| `src/app.rs`            | `run_watch` coordinator + writer threads, `WatchHandle`.             |
| `src/tui.rs`            | Ratatui interactive search loop.                                     |
| `src/main.rs`           | TUI launcher; clap-parsed arguments + `run_watch` + `run_tui`.       |

## Threading model

`run_watch` (long-lived; the only entry point now):

```
Scanner (startup pass)
   └── extractor pool ──► flume ──► DB writer thread
                                       └── overlay.modify_doc / tombstone_doc
                                              under a single RwLock write guard
notify-debouncer-full thread
   └── flume::Sender<WatchEvent> ──► coordinator thread
                                          ├── Dirty(path):  scan_one → extractor
                                          └── Removed(path): writer Delete
```

Synchronization primitives, exactly as the report names them:

* `arc-swap::ArcSwap<BaseIndex>` for the atomic base-index swap.
* `parking_lot::RwLock<Overlay>` for the small mutable overlay.
* `flume::bounded` for `ScanJob` / `ExtractedDoc` / `WriterMsg` / `WatchEvent`.
* `AtomicU64::fetch_or` inside the bigram builder for race-free dense writes
  without `unsafe`.

## Query algorithm

```rust
fn literal_search(state, query, limit) -> Vec<Hit> {
    let q = normalize_query_ascii(query);
    let needle = q.as_bytes();
    let finder = memchr::memmem::Finder::new(needle);

    let base = state.load_base();
    let ov   = state.overlay.read();

    // Candidate generation
    let mut cand = base.bigrams.and_then(|b| b.query(needle));
    if let Some(bits) = &mut cand {
        // hide stale base chunks
        for w in 0..bits.len() { bits[w] &= !ov.tombstones[w]; }
    }

    // Verification: base
    for i in base.chunks.iter().enumerate().filter(in_cand_or_full) {
        if let Some(pos) = finder.find(&chunk.text_norm_ascii) {
            hits.push(make_hit(chunk, pos, needle.len()));
        }
    }

    // Verification: overlay
    let qb = extract_bigrams(needle);
    for i in ov.overflow_matches(&qb) {
        if let Some(pos) = finder.find(&ov.overflow_chunks[i].text_norm_ascii) {
            hits.push(make_hit(...));
        }
    }
    cheap_rank(&mut hits, &q);
    hits.truncate(limit);
    hits
}
```

Regex and fuzzy follow the same shape: `regex_to_bigram_query(pattern)` /
`fuzzy_to_bigram_query(pattern, 6)` produces a `BigramQuery` tree;
`evaluate(&base.bigrams)` yields a candidate bitset; verification compiles
the pattern with `regex::RegexBuilder::new(p).case_insensitive(true)` or
ranks with `neo_frizbee::match_list_parallel`.

## Rebuild

The overlay is bounded by two thresholds (constants in `src/index.rs`):

* `REBUILD_OVERLAY_CHUNKS = 10_000` — collapse when the overlay holds this
  many overflow chunks.
* `REBUILD_TOMBSTONE_RATIO = 0.10` — collapse when more than 10% of base
  chunks are tombstoned.

The DB writer thread checks `IndexState::needs_rebuild` after every
overlay-mutating message. When it trips, `rebuild_from_db` streams a fresh
`BaseIndex` out of SQLite and publishes it with a single `ArcSwap::store`;
the overlay is reset under the same write guard. Readers never see a torn
snapshot.

## Tests

* `cargo test` covers the unit tests plus `tests/watch_pipeline.rs`,
  which exercises scan → extract → overlay → search end-to-end across
  initial scan, new arrivals, modifications, and deletions.
* `benches/literal_search.rs` and `benches/bigram_build.rs` provide
  criterion benchmarks scaled to illustrate the path to the report's
  100k-chunk target.

## Attribution notes

* `src/bigram.rs` and `src/bigram_query.rs` carry SPDX MIT headers with the
  upstream attribution and a list of the pdffff-specific modifications
  (`AtomicU64::fetch_or` instead of `UnsafeCell` slab; `Vec<u8>` instead of
  `smallvec::SmallVec`).
* The chunked-page schema and the `pdftotext - -` extraction shape are
  taken from the report's description of `rga`'s built-in poppler adapter.
* `neo_frizbee` is the same fuzzy scorer `fff` uses; no rewrite, no fork.
