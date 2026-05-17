//! Criterion benchmark for `build_bigram_index_from_chunks`.
//!
//! The bigram build is what dominates *startup* latency for a corpus
//! loaded from SQLite. The numbers reported here translate to the
//! report's "1,000 mixed PDFs" criterion as follows: roughly 25 chunks
//! per PDF means 1,000 PDFs is ~25,000 chunks, which lands between the
//! `10_000` and `50_000` rows here. We pick the same N values as the
//! query benchmark so the two reports line up.
//!
//! Run with: `cargo bench --bench bigram_build`.

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use pdffff::bigram::build_bigram_index_from_chunks;
use pdffff::index::ChunkItem;

const VOCAB: &[&str] = &[
    "the", "of", "and", "to", "in", "is", "for", "that", "with", "as",
    "method", "algorithm", "result", "theorem", "lemma", "proof",
    "figure", "table", "section", "chapter", "abstract", "introduction",
    "conclusion", "references", "bibliography", "appendix", "function",
    "variable", "matrix", "vector", "tensor", "graph", "node", "edge",
    "tree", "leaf", "root", "search", "index", "query", "result",
    "document", "page", "chunk", "token", "bigram", "trigram", "filter",
];

fn make_chunks(n: usize) -> Vec<ChunkItem> {
    let mut rng = ChaCha8Rng::seed_from_u64(0xc0ffee);
    (0..n)
        .map(|i| {
            let words: Vec<&str> = (0..60)
                .map(|_| *VOCAB.choose(&mut rng).expect("vocab nonempty"))
                .collect();
            let text = words.join(" ");
            let norm = pdffff::normalize::normalize_for_index(&text);
            ChunkItem::new(
                i as i64,
                (i / 25) as i64,
                Arc::from("/synth/c.pdf"),
                Arc::from("c.pdf"),
                (i % 25 + 1) as u32,
                (i % 25) as u32,
                0,
                text.len() as u32,
                Arc::<str>::from(text.as_str()),
                Arc::<[u8]>::from(norm.as_bytes()),
                Arc::<str>::from(text.as_str()),
                0,
            )
        })
        .collect()
}

fn bench_bigram_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bigram_build");
    for &n in &[1_000usize, 10_000, 50_000] {
        let chunks = make_chunks(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &chunks, |b, chunks| {
            b.iter(|| {
                let idx = build_bigram_index_from_chunks(chunks);
                std::hint::black_box(idx);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bigram_build);
criterion_main!(benches);
