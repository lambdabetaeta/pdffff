//! Shared helpers for Day-2 integration tests.
//!
//! We synthesize tiny PDFs on the fly using `printpdf` so the test corpus
//! is part of the test source, not a checked-in binary asset. The
//! resulting PDFs are saved to a tempdir, then fed through the real
//! `pdftotext` binary by the extractor pipeline — i.e., this is an
//! end-to-end test, not a unit test.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use printpdf::{BuiltinFont, Mm, PdfDocument};

/// Build a 2-page PDF using the built-in Times-Roman font and write it
/// to `path`. Each page contains the supplied `text` (split on `|` into
/// separate `use_text` calls so we don't depend on automatic wrapping).
pub fn make_pdf_two_pages(path: &Path, page1_text: &[&str], page2_text: &[&str]) {
    let (doc, page1, layer1) =
        PdfDocument::new("pdffff test fixture", Mm(210.0), Mm(297.0), "Layer 1");
    let font = doc.add_builtin_font(BuiltinFont::TimesRoman).unwrap();
    let layer = doc.get_page(page1).get_layer(layer1);
    let mut y = 280.0;
    for line in page1_text {
        layer.use_text(*line, 14.0, Mm(20.0), Mm(y), &font);
        y -= 10.0;
    }

    let (page2_idx, layer2_idx) = doc.add_page(Mm(210.0), Mm(297.0), "Page 2 Layer");
    let layer2 = doc.get_page(page2_idx).get_layer(layer2_idx);
    let mut y = 280.0;
    for line in page2_text {
        layer2.use_text(*line, 14.0, Mm(20.0), Mm(y), &font);
        y -= 10.0;
    }

    let f = File::create(path).expect("create pdf path");
    doc.save(&mut BufWriter::new(f)).expect("save pdf");
}

/// Build a PDF that has pages but no visible text content. `pdftotext`
/// extracts only a trailing form-feed for such PDFs, yielding empty /
/// whitespace-only output — i.e., `DocStatus::Empty`.
pub fn make_pdf_no_text(path: &Path) {
    let (doc, _page, _layer) =
        PdfDocument::new("empty fixture", Mm(210.0), Mm(297.0), "Layer 1");
    let f = File::create(path).expect("create pdf path");
    doc.save(&mut BufWriter::new(f)).expect("save pdf");
}
