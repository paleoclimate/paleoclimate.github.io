# The viewer has no age slider

Each published map is StatiCrypt ciphertext. Dragging the Ancient–Recent range loaded a new map on the way through, and decrypting those pages froze the atlas when the years moved quickly. The range, its tick marks, and the Recent and Ancient labels are gone. What remained was the age menu, the younger and older buttons, and the left and right arrow keys. The buttons and those arrow keys were removed later; see `docs/adr/0038-viewer-age-changes-only-through-the-menu.md`.

**Considered options:** (1) remove the slider; (2) keep it and decrypt only the age where the drag ends. (2) still leaves a control whose job is to sweep ages, which is the gesture that stalled the page.
