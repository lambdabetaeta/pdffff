//! Day-2 placeholder: PDF extraction via `pdftotext`. The real
//! implementation lives in `src/extract.rs` after the Day-2 agent runs.

#![allow(dead_code)]

use anyhow::Result;
use std::path::Path;

use crate::db::ExtractedDoc;
use crate::scanner::ScanJob;

/// Placeholder. Replaced on Day 2.
pub fn extract_pdf(_job: &ScanJob) -> Result<ExtractedDoc> {
    unimplemented!("extract_pdf: implemented on Day 2")
}

/// Placeholder; concrete page-splitting + chunking lands on Day 2.
pub fn split_pages(_text: &str) -> Vec<&str> {
    unimplemented!("split_pages: implemented on Day 2")
}

#[allow(dead_code)]
fn _unused_path(_p: &Path) {}
