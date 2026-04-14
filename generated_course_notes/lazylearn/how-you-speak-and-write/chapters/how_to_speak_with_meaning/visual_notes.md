# Visual Evidence
## Frame Inventory
- `how_to_speak_with_meaning_figure_02.png`: A pale yellow slide showing a central red circle labeled `@` surrounded by six purple circles in a symmetric audience-like arrangement; this screenshot should remain in the final notes.
- `how_to_speak_with_meaning_figure_03.png`: A dark magenta slide showing a top question box, a downward arrow into a yellow decision diamond, and two branches ending in a magenta X-like box and a blue check-mark box; this screenshot should remain in the final notes.
- `how_to_speak_with_meaning_figure_04.png`: A light lavender slide showing abstract paragraph-like lines with two red underline marks indicating selected words; this screenshot should remain in the final notes.
- `how_to_speak_with_meaning_figure_05.png`: A cyan slide showing a simplified standing figure with a blue equals sign between the legs to indicate balance; this screenshot should remain in the final notes.

## Equation Extraction
- `how_to_speak_with_meaning_figure_02.png`
  - [visible] \(\text{@}\)
  - [standard completion] \(\text{speaker/message} \longrightarrow \text{audience members}\)
- `how_to_speak_with_meaning_figure_03.png`
  - [visible] \(\text{?}\)
  - [partially visible] \(\times\) for the left endpoint symbol
  - [visible] \(\checkmark\) for the right endpoint symbol
- `how_to_speak_with_meaning_figure_04.png`
  - [visible] \(\underline{\phantom{\text{word}}}\)
  - [standard completion] \(\underline{\text{key word}}\)
- `how_to_speak_with_meaning_figure_05.png`
  - [visible] \(=\)
  - [standard completion] \(\text{weight on left foot} = \text{weight on right foot}\)

## Diagram Extraction
- `how_to_speak_with_meaning_figure_02.png` is a radial node-layout diagram. It should be shown both as the original screenshot and as a nearby TikZ redraw with one central labeled node and six surrounding nodes.
- `how_to_speak_with_meaning_figure_03.png` is a top-down decision-tree sketch. It should be shown both as the original screenshot and as a nearby TikZ redraw preserving the question node, decision diamond, two outgoing branches, and contrasting outcomes.
- `how_to_speak_with_meaning_figure_04.png` is an abstract markup diagram rather than a readable text excerpt. It should be shown both as the original screenshot and as a stylized TikZ redraw of paragraph lines with selected underline marks.
- `how_to_speak_with_meaning_figure_05.png` is a posture-and-balance sketch. It should remain as a screenshot and be paired nearby with a cleaned displayed equation; a TikZ redraw is optional rather than necessary.

## Reconstruction Guidance
- Keep all four screenshots visible in the final notes, because each one carries visual structure that a pure prose summary would flatten.
- For `how_to_speak_with_meaning_figure_02.png`, reconstruct only the structural idea that a central message source addresses multiple audience nodes. Do not over-annotate the redraw with extra labels unless the transcript explicitly supplies them.
- For `how_to_speak_with_meaning_figure_03.png`, reconstruct the branching logic cleanly in TikZ, but treat the branch meanings cautiously. The image gives us structure and contrast, not a fully labeled logical derivation.
- For `how_to_speak_with_meaning_figure_04.png`, reconstruct the idea of paragraph emphasis rather than literal text. The image is evidence for markup practice, not for any specific sentence.
- For `how_to_speak_with_meaning_figure_05.png`, pair the screenshot with a displayed equation such as
  \[
  \text{weight on left foot} = \text{weight on right foot}.
  \]
  This should be presented as a transcript-backed clean formulation of the visible balance symbol, not as a literal on-screen equation.
- In all cases, prefer minimal reconstructions that clarify the lecture’s procedure over elaborate diagrams that introduce information not present in the frames.

## Uncertainties
- In `how_to_speak_with_meaning_figure_02.png`, the `@` symbol is fully visible, but its exact meaning is not visually spelled out; the interpretation as speaker, message, or email-script source comes from the nearby transcript.
- In `how_to_speak_with_meaning_figure_03.png`, the left endpoint symbol is stylized and not perfectly legible as a mathematical `\times`; it is safest to describe it as an X-like reject mark.
- In `how_to_speak_with_meaning_figure_03.png`, the branches are unlabeled, so any identification of them with specific lecture categories should be transcript-backed rather than image-only.
- In `how_to_speak_with_meaning_figure_04.png`, no actual words are readable. Any underlined “key word” written in the notes is necessarily a standard completion of the visible emphasis pattern.
- In `how_to_speak_with_meaning_figure_05.png`, only the equals sign is literally visible. The full statement about left-foot and right-foot weight is a cautious completion from the spoken instruction.