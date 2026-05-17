//! In-memory chunk index.
//!
//! Three cooperating sub-modules:
//!
//! * [`chunk`] — [`ChunkItem`], the row type the indexer publishes.
//! * [`base`] — the immutable [`BaseIndex`] built from SQLite plus the
//!   dense bigram prefilter over it; one DB streaming pass routed
//!   through [`load_base_index_from_db`].
//! * [`overlay`] — the [`Overlay`] applied on top of the base index
//!   between rebuilds, together with [`OverflowSet`] (the chunk-
//!   publishing side) and the rebuild thresholds.
//! * [`state`] — [`IndexState`], the handle the rest of the system
//!   reads from, and the [`rebuild_from_db`] routine.
//!
//! Two consistency rules are load-bearing for query correctness:
//!
//! 1. **No double-visibility.** When a doc is modified, the tombstone
//!    must hide the doc's base range *before* the overflow rows
//!    become readable, so a reader never sees both the stale and the
//!    fresh chunks for the same doc. [`Overlay::modify_doc`] performs
//!    both mutations under a single `RwLock` write guard.
//! 2. **Snapshot reads.** A query holds `state.load_base()` *and* a
//!    single read guard on `state.overlay` for the entire literal
//!    pass.

pub mod base;
pub mod chunk;
pub mod overlay;
pub mod state;

pub use base::{BaseIndex, load_base_index_from_db};
pub use chunk::ChunkItem;
pub use overlay::{
    Overlay, OverflowSet, OverlayStats, REBUILD_OVERLAY_CHUNKS, REBUILD_TOMBSTONE_RATIO,
};
pub use state::{IndexState, rebuild_from_db};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigram::extract_bigrams;
    use crate::db::{ChunkInsert, Db, DocStatus, ExtractedDoc};
    use crate::normalize::{NORM_VERSION, normalize_for_index};
    use anyhow::Result;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn fake_doc(path: PathBuf, pages: &[(&str, u32)]) -> ExtractedDoc {
        let chunks: Vec<ChunkInsert> = pages
            .iter()
            .enumerate()
            .map(|(i, (text, page_no))| ChunkInsert {
                page_no: *page_no,
                chunk_ord: i as u32,
                char_start: 0,
                char_end: text.len() as u32,
                text_utf8: (*text).to_string(),
                text_norm_ascii: normalize_for_index(text),
                preview: (*text).to_string(),
            })
            .collect();
        ExtractedDoc {
            path,
            size_bytes: 0,
            mtime_ns: 0,
            dev: None,
            ino: None,
            extractor: "test".into(),
            extractor_version: "test".into(),
            norm_version: NORM_VERSION,
            page_count: pages.len() as u32,
            status: DocStatus::Ok,
            error_text: None,
            chunks,
        }
    }

    fn mk_chunk(id: i64, doc: i64, text: &str) -> ChunkItem {
        ChunkItem::new(
            id,
            doc,
            Arc::from("/x.pdf"),
            Arc::from("x.pdf"),
            1,
            0,
            0,
            text.len() as u32,
            Arc::from(text),
            Arc::<[u8]>::from(normalize_for_index(text).as_bytes()),
            Arc::from(text),
            0,
        )
    }

    fn base_with_three_docs() -> BaseIndex {
        // Three docs, two chunks each, in (doc_id, chunk_ord) order.
        let chunks: Vec<ChunkItem> = vec![
            mk_chunk(1, 1, "alpha bravo charlie"),
            mk_chunk(2, 1, "delta echo foxtrot"),
            mk_chunk(3, 2, "golf hotel india"),
            mk_chunk(4, 2, "juliet kilo lima"),
            mk_chunk(5, 3, "mike november oscar"),
            mk_chunk(6, 3, "papa quebec romeo"),
        ];
        let mut doc_ranges = HashMap::new();
        doc_ranges.insert(1, 0..2);
        doc_ranges.insert(2, 2..4);
        doc_ranges.insert(3, 4..6);
        BaseIndex {
            chunks: Arc::new(chunks),
            doc_ranges,
            bigrams: None,
            filename_norms: HashMap::new(),
            built_at_ms: 0,
        }
    }

    #[test]
    fn empty_db_loads_to_empty_index() -> Result<()> {
        let tmp = tempdir()?;
        let db = Db::open(&tmp.path().join("idx.db"))?;
        let base = load_base_index_from_db(&db)?;
        assert!(base.chunks.is_empty());
        assert!(base.doc_ranges.is_empty());
        // No chunks → no prefilter is built (full-scan over the
        // empty Vec is trivial).
        assert!(base.bigrams.is_none());
        Ok(())
    }

    #[test]
    fn populated_db_attaches_bigram_index() -> Result<()> {
        let tmp = tempdir()?;
        let db_path = tmp.path().join("idx.db");
        let mut db = Db::open(&db_path)?;
        db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/a.pdf"),
            &[("alpha bravo charlie", 1)],
        ))?;
        drop(db);

        let db = Db::open(&db_path)?;
        let base = load_base_index_from_db(&db)?;
        let bigrams = base.bigrams.as_ref().expect("bigrams attached");
        assert_eq!(bigrams.populated(), base.chunks.len());
        assert_eq!(bigrams.item_count(), base.chunks.len());
        Ok(())
    }

    #[test]
    fn loads_doc_ranges_in_order() -> Result<()> {
        let tmp = tempdir()?;
        let db_path = tmp.path().join("idx.db");
        let mut db = Db::open(&db_path)?;
        let id_a = db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/a.pdf"),
            &[("alpha bravo", 1), ("charlie delta", 2)],
        ))?;
        let id_b = db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/b.pdf"),
            &[("echo foxtrot", 1)],
        ))?;
        drop(db);

        let db = Db::open(&db_path)?;
        let base = load_base_index_from_db(&db)?;
        assert_eq!(base.chunks.len(), 3);
        let range_a = base.doc_ranges.get(&id_a).expect("doc a present");
        let range_b = base.doc_ranges.get(&id_b).expect("doc b present");
        assert_eq!(range_a.len(), 2);
        assert_eq!(range_b.len(), 1);
        // doc_id is monotone in insertion order, so ranges should be
        // contiguous and ordered.
        assert_eq!(range_a.end, range_b.start);

        // Path is shared per doc — pointer-equal across chunks of the
        // same document, distinct across documents.
        let a0 = &base.chunks[range_a.start];
        let a1 = &base.chunks[range_a.start + 1];
        let b0 = &base.chunks[range_b.start];
        assert!(Arc::ptr_eq(&a0.path, &a1.path));
        assert!(!Arc::ptr_eq(&a0.path, &b0.path));
        assert_eq!(&*a0.filename, "a.pdf");
        assert_eq!(&*b0.filename, "b.pdf");

        // Norm bytes are exactly what the normalizer produced.
        assert_eq!(&*a0.text_norm_ascii, b"alpha bravo");
        assert_eq!(&*b0.text_norm_ascii, b"echo foxtrot");
        Ok(())
    }

    #[test]
    fn deleted_doc_chunks_are_excluded() -> Result<()> {
        let tmp = tempdir()?;
        let db_path = tmp.path().join("idx.db");
        let mut db = Db::open(&db_path)?;
        db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/keep.pdf"),
            &[("kept text", 1)],
        ))?;
        db.upsert_extracted(&fake_doc(
            PathBuf::from("/tmp/drop.pdf"),
            &[("dropped text", 1)],
        ))?;
        db.mark_deleted(std::path::Path::new("/tmp/drop.pdf"))?;
        drop(db);

        let db = Db::open(&db_path)?;
        let base = load_base_index_from_db(&db)?;
        assert_eq!(base.chunks.len(), 1);
        assert_eq!(&*base.chunks[0].text_norm_ascii, b"kept text");
        Ok(())
    }

    #[test]
    fn overlay_tombstone_doc_hides_base_range() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        assert!(!ov.is_tombstoned(0));
        ov.tombstone_doc(2, &base);
        // Doc 2 occupies indices 2..4 — both must be tombstoned.
        assert!(ov.is_tombstoned(2));
        assert!(ov.is_tombstoned(3));
        // Other docs untouched.
        assert!(!ov.is_tombstoned(0));
        assert!(!ov.is_tombstoned(1));
        assert!(!ov.is_tombstoned(4));
        assert!(!ov.is_tombstoned(5));
        assert!(ov.changed_docs.contains(&2));
        // Tombstoning bumps generation.
        assert!(ov.generation >= 1);
    }

    #[test]
    fn overlay_tombstone_doc_drops_prior_overflow() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        // First publish two overflow rows for doc 2 (simulating a
        // prior modify).
        ov.add_overflow(mk_chunk(100, 2, "new doc 2 chunk one"));
        ov.add_overflow(mk_chunk(101, 2, "new doc 2 chunk two"));
        // Also publish one for doc 3 — must survive.
        ov.add_overflow(mk_chunk(200, 3, "doc 3 overflow"));
        assert_eq!(ov.overflow.len(), 3);

        // Now tombstone doc 2 — its previous overflow rows must go,
        // doc 3's must stay.
        ov.tombstone_doc(2, &base);
        assert_eq!(ov.overflow.len(), 1);
        assert_eq!(ov.overflow.chunks()[0].doc_id, 3);
        // Doc 2's base range must now be tombstoned.
        assert!(ov.is_tombstoned(2));
        assert!(ov.is_tombstoned(3));
    }

    #[test]
    fn overlay_modify_doc_replaces_atomically() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        // Doc 1's base range (indices 0..2) is initially visible.
        assert!(!ov.is_tombstoned(0));
        assert!(!ov.is_tombstoned(1));
        let new_chunks = vec![
            mk_chunk(500, 1, "fresh doc 1 chunk one"),
            mk_chunk(501, 1, "fresh doc 1 chunk two"),
        ];
        ov.modify_doc(1, new_chunks, &base);
        // Base hidden:
        assert!(ov.is_tombstoned(0));
        assert!(ov.is_tombstoned(1));
        // Overflow now carries the two new chunks for doc 1, in
        // insertion order.
        assert_eq!(ov.overflow.len(), 2);
        assert_eq!(ov.overflow.chunks()[0].chunk_id, 500);
        assert_eq!(ov.overflow.chunks()[1].chunk_id, 501);
        // And `changed_docs` lists the touched doc.
        assert!(ov.changed_docs.contains(&1));
    }

    #[test]
    fn overlay_modify_doc_replays_replace_themselves() {
        // A doc modified twice must show only the second set of
        // chunks afterwards — the first modify's overflow rows must
        // be dropped by the second call.
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());

        let first = vec![mk_chunk(700, 3, "first revision")];
        ov.modify_doc(3, first, &base);
        assert_eq!(ov.overflow.len(), 1);
        assert_eq!(ov.overflow.chunks()[0].chunk_id, 700);

        let second = vec![
            mk_chunk(701, 3, "second revision one"),
            mk_chunk(702, 3, "second revision two"),
        ];
        ov.modify_doc(3, second, &base);
        assert_eq!(ov.overflow.len(), 2);
        assert!(ov.overflow.chunks().iter().all(|c| c.doc_id == 3));
        let ids: Vec<i64> = ov.overflow.chunks().iter().map(|c| c.chunk_id).collect();
        assert_eq!(ids, vec![701, 702]);
    }

    #[test]
    fn overlay_overflow_matches_filters_by_bigrams() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        // Three overflow rows with distinctive token tails.
        ov.add_overflow(mk_chunk(10, 1, "the quick brown fox jumps"));
        ov.add_overflow(mk_chunk(11, 2, "lazy dog watches the river"));
        ov.add_overflow(mk_chunk(12, 3, "the quick zebra"));

        // Query bigrams for "quick" → "qu", "ui", "ic", "ck"
        let q = extract_bigrams(b"quick");
        let hits = ov.overflow_matches(&q);
        // Should be indices 0 and 2 (both contain "quick"), in order.
        assert_eq!(hits, vec![0, 2]);

        // A bigram set with a non-existent bigram → no matches.
        let q = extract_bigrams(b"zzzz");
        assert_eq!(ov.overflow_matches(&q), Vec::<usize>::new());

        // Empty query → every overflow row.
        assert_eq!(ov.overflow_matches(&[]), vec![0, 1, 2]);
    }

    #[test]
    fn overlay_clear_resets_everything() {
        let base = base_with_three_docs();
        let mut ov = Overlay::new(base.chunks.len());
        ov.add_overflow(mk_chunk(10, 1, "anything"));
        ov.tombstone_doc(2, &base);
        assert!(!ov.is_empty());
        ov.clear();
        assert!(ov.overflow.is_empty());
        assert!(ov.changed_docs.is_empty());
        assert_eq!(ov.generation, 0);
        // After `clear`, the tombstone bitset is empty — the next
        // base index will reset it via `Overlay::new`. Until then no
        // index is in range.
        for i in 0..base.chunks.len() {
            assert!(!ov.is_tombstoned(i));
        }
    }

    #[test]
    fn needs_rebuild_threshold_triggers() {
        // Tombstone enough of a tiny base to cross the ratio.
        let base = base_with_three_docs();
        let state = IndexState::new(base);
        let ov = state.overlay.read();
        let base = state.load_base();
        assert!(!state.needs_rebuild(&ov, &base));
        drop(ov);

        // Tombstone the first doc (2 of 6 chunks = 33%, > 10%).
        let base = state.load_base();
        {
            let mut ov = state.overlay.write();
            ov.tombstone_doc(1, &base);
        }
        let ov = state.overlay.read();
        assert!(state.needs_rebuild(&ov, &base));
    }
}
