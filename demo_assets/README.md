# Synthetic demo images

`synthetic_checker.pgm` and `synthetic_flat.pgm` are tiny, project-created pixel
patterns for exercising the upload interface. They contain no organ-on-a-chip,
patient, or dataset imagery. Upload the checker to observe a `PASS` under the
uncalibrated rules, and the flat image to observe a `REACQUIRE` suggestion.
This is a deliberate reminder that a passing rule does **not** establish that
an image contains a valid biological sample or has a good expert label.

The files use ASCII PGM (`P2`), which Pillow can open without extra codecs.

