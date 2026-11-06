# frozen_string_literal: true

require "rouge"

module Rouge
  module Lexers
    class Lean
      # The upstream lexer targets Lean 3 and marks these valid symbols as errors.
      # Keep the article's Lean 4 source intact and reuse the theme's token styles.
      prepend :root do
        rule %r/[{}·]/, Punctuation
      end
    end
  end
end
