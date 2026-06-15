// Harmonic Noise Suppression — Typst Report Template
// Based on: @preview/starter-journal-article:0.5.1
// Customisations:
//   - No title page (content starts immediately on first page)
//   - Abstract + TOC included
//   - Header with project name + report title
//   - Informal tone (less ceremony than a journal paper)

#import "@preview/starter-journal-article:0.5.1": article, author-meta

// Re-export for consumers
#let author-meta = author-meta

#let report(
  title: "Report Title",
  authors: ("Author Name": author-meta("project")),
  affiliations: ("project": "Harmonic Noise Suppression Project"),
  abstract: [],
  keywords: (),
  date: none,
  body,
) = {
  // Use the base article template
  show: article.with(
    title: title,
    authors: authors,
    affiliations: affiliations,
    abstract: abstract,
    keywords: keywords,
  )

  // Custom header
  set page(
    header: context {
      let page-num = here().page()
      if page-num > 1 {
        align(right)[
          #smallcaps[HNS] — #title #h(1fr) #page-num
        ]
      }
    },
  )

  // Table of contents on first page
  if abstract != [] {
    pagebreak(weak: true)
  }
  outline(depth: 2, title: [Contents])
  pagebreak(weak: true)

  body
}
