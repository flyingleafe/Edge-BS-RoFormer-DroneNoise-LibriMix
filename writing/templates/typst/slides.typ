// Harmonic Noise Suppression — Typst Slides Template
// Framework: Touying (https://touying-typ.github.io/)
// Theme: simple (https://touying-typ.github.io/docs/themes/simple)

#import "@preview/touying:0.6.1": *
#import themes.simple: *

#let hns-slide(
  config: (:),
  repeat: auto,
  setting: body => body,
  composer: auto,
  ..bodies,
) = {
  touying-slide-wrapper(self => {
    // Footer with page numbers
    let footer = self => {
      set align(bottom)
      set text(size: 0.8em)
      show: pad.with(.5em)
      components.left-and-right(
        none,
        context text(fill: gray, utils.slide-counter.display() + " / " + utils.last-slide-number),
      )
    }

    let self = utils.merge-dicts(
      self,
      config-page(
        header: none,
        footer: footer,
      ),
      config-common(subslide-preamble: none),
    )

    // Get heading from the current section
    let heading = utils.display-current-heading(level: 1, depth: self.slide-level)
    let heading-block = block(
      below: 1.5em,
      text(1.2em, weight: "bold", heading)
    )

    // Wrap the user's setting to prepend heading
    let new-setting = body => {
      setting(heading-block + body)
    }

    touying-slide(self: self, config: config, repeat: repeat, setting: new-setting, composer: composer, ..bodies)
  })
}

#let hns-slides(
  title: "Presentation Title",
  subtitle: none,
  author: none,
  date: none,
  aspect-ratio: "16-9",
  body,
) = {
  // Custom init: smaller font, left-aligned, tighter spacing
  let custom-init(self: none, body) = {
    set text(size: 16pt)
    set align(left)
    set par(leading: 0.5em)
    show footnote.entry: set text(size: .5em)
    show heading.where(level: 1): set text(1.1em)
    show figure: set text(size: 0.6em)
    show math.equation: set text(size: 0.9em)
    body
  }

  // Use the simple theme with our customisations
  show: simple-theme.with(
    aspect-ratio: aspect-ratio,
    config-common(
      slide-fn: hns-slide,
      new-section-slide-fn: none,
    ),
    config-methods(
      init: custom-init,
      alert: utils.alert-with-primary-color,
    ),
  )

  // Title slide using the theme's title-slide
  title-slide[
    #align(center + horizon)[
      #block[
        #text(2em, weight: "bold", title)
        #if subtitle != none {
          linebreak()
          text(1.2em, subtitle)
        }
        #if author != none {
          linebreak()
          text(1em, author)
        }
        #if date != none {
          linebreak()
          text(0.9em, date)
        }
      ]
    ]
  ]

  body
}
