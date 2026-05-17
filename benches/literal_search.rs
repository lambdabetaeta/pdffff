//! Criterion benchmark for `pdffff::query::search` in literal mode.
//!
//! Synthesises a deterministic English-ish corpus of N chunks, builds a
//! `BaseIndex` (with bigram prefilter), then runs a fixed list of
//! representative queries through `search(..., QueryMode::Literal, ..)`.
//! Throughput is reported in `elements/iter` over the query list.
//!
//! Translation to the report's success criterion ("≤ 50 ms p95 at 100k
//! chunks"):
//!
//! * `chunks/50000` here is half the target corpus size. If a literal
//!   query lands at ~Y ms on this machine for 50k chunks, then 100k
//!   chunks would land near `2 * Y` ms in the linear-scan regime and
//!   roughly the same `Y` ms when the bigram prefilter actually
//!   eliminates most candidates (because the verifier-bound term
//!   dominates). Both regimes are exercised below: `the` triggers
//!   the ubiquity drop (no prefilter info, full scan) while
//!   `xylotomous` survives as a rare needle (prefilter narrows to a
//!   handful of chunks).
//!
//! Run with: `cargo bench --bench literal_search`.

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use pdffff::index::{BaseIndex, ChunkItem, IndexState};
use pdffff::query::{QueryMode, search};

const VOCAB: &[&str] = &[
    "the", "of", "and", "to", "in", "is", "for", "that", "with", "as",
    "on", "by", "an", "this", "we", "are", "be", "from", "not", "or",
    "have", "at", "but", "they", "which", "one", "you", "all", "any",
    "may", "would", "should", "their", "these", "those", "system",
    "method", "algorithm", "result", "theorem", "lemma", "proof",
    "figure", "table", "section", "chapter", "abstract", "introduction",
    "conclusion", "references", "bibliography", "appendix", "function",
    "variable", "matrix", "vector", "tensor", "graph", "node", "edge",
    "tree", "leaf", "root", "search", "index", "query", "result",
    "document", "page", "chunk", "token", "bigram", "trigram", "filter",
    "candidate", "match", "score", "rank", "snippet", "context",
];

const RARE_NEEDLE: &str = "xylotomous"; // appears in exactly one chunk

const QUERIES: &[&str] = &[
    "search",         // common: density filter probably drops it
    "the algorithm",  // phrase, two common terms
    "function variable",
    "snippet context",
    "xylotomous",     // rare, prefilter narrows hard
];

/// Build a chunk's content: 60 random words from `VOCAB`, plus the
/// rare needle in exactly one chunk (chunk index 0) so the prefilter
/// has something to find.
fn make_chunk(rng: &mut ChaCha8Rng, idx: usize, total_chunks: usize) -> ChunkItem {
    let mut words: Vec<&str> = (0..60)
        .map(|_| *VOCAB.choose(rng).expect("vocab nonempty"))
        .collect();
    if idx == 0 {
        words.insert(30, RARE_NEEDLE);
    }
    // Sprinkle the rare needle into a few specific chunks as well so
    // searches for it find more than one hit even with very large N.
    if idx % (total_chunks.max(1) / 10).max(1) == 7 {
        words.insert(40, RARE_NEEDLE);
    }
    let text = words.join(" ");
    let norm = pdffff::normalize::normalize_for_index(&text);
    let path = "/synth/chunk.pdf";
    ChunkItem::new(
        idx as i64,
        (idx / 25) as i64, // ~25 chunks per "doc"
        Arc::from(path),
        Arc::from("chunk.pdf"),
        (idx % 25 + 1) as u32,
        (idx % 25) as u32,
        0,
        text.len() as u32,
        Arc::<str>::from(text.as_str()),
        Arc::<[u8]>::from(norm.as_bytes()),
        Arc::<str>::from(text.as_str()),
        0,
    )
}

fn build_state(n: usize) -> IndexState {
    let mut rng = ChaCha8Rng::seed_from_u64(0xdeadbeef);
    let chunks: Vec<ChunkItem> = (0..n).map(|i| make_chunk(&mut rng, i, n)).collect();
    let mut doc_ranges = std::collections::HashMap::new();
    let mut cur: Option<(i64, usize)> = None;
    for (i, c) in chunks.iter().enumerate() {
        if cur.map(|(d, _)| d) != Some(c.doc_id) {
            if let Some((d, s)) = cur {
                doc_ranges.insert(d, s..i);
            }
            cur = Some((c.doc_id, i));
        }
    }
    if let Some((d, s)) = cur {
        doc_ranges.insert(d, s..chunks.len());
    }
    let bigrams = if chunks.is_empty() {
        None
    } else {
        Some(Arc::new(pdffff::bigram::build_bigram_index_from_chunks(
            &chunks,
        )))
    };
    let base = BaseIndex {
        chunks: Arc::new(chunks),
        doc_ranges,
        bigrams,
        filename_norms: std::collections::HashMap::new(),
        built_at_ms: 0,
    };
    IndexState::new(base)
}

fn bench_literal(c: &mut Criterion) {
    let mut group = c.benchmark_group("literal_search");
    for &n in &[1_000usize, 10_000, 50_000] {
        let state = build_state(n);
        group.throughput(Throughput::Elements(QUERIES.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &state, |b, state| {
            b.iter(|| {
                for q in QUERIES {
                    let hits = search(state, q, QueryMode::Literal, 100).expect("search");
                    std::hint::black_box(hits);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_literal);
criterion_main!(benches);
