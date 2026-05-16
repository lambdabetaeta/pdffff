//! Day-3 placeholder: search engine over `BaseIndex` + `Overlay`.

#![allow(dead_code)]

pub struct Hit {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub path: String,
    pub page_no: u32,
    pub score: f32,
    pub snippet: String,
}
