# Walkthrough visual-accuracy spot-check (hand-rated)

Visual accuracy is NOT auto-measured - there are no labels for whether
the model read a photo correctly. Rate each observation against its
photo: mark [x] when accurate; leave unchecked and add a note when not.

Rater: Claude (Fable 5) vision review, 2026-07-21 - an independent-model
cross-check of the Haiku observations, per-photo against the source image.
Not a field-professional rating; override any line on human review.

**Result: 17/30 observations strictly accurate.** The errors are systematic,
not random: (a) ~8 over-calls of hairline cracking / mineral deposits on
clean textured concrete (dataset label no_defect) - the false-positive
direction, consistent with the Phase 5.4 OOD finding; (b) 3 misreads of
degenerate or out-of-distribution crops (a near-black crop, a drawn/graffiti
figure, precast formwork grooves); (c) 2 partially-right observations.
Caveat: the golden set is built from 256px dataset crops that are far harder
than real technician photos (no context, no scale); treat these numbers as a
lower bound and the "verify on-site" framing as load-bearing.

## walk_01
- [x] photo_1 -> RATED NOT ACCURATE: the faint line reads as a surface scuff/joint, not a hairline crack; no efflorescence discernible (dataset label: no_defect). Original claim: Vertical hairline crack visible on concrete surface with subtle discoloration along its length suggesting efflorescence or moisture staining. [ ]
- [x] photo_2 (inachi-012, eng-014, hud-012): Concrete spalling and surface deterioration with chunks noted by technician as found on floor below this area. Light scaling and flaking visible on the concrete surface.
- [ ] photo_3 (hud-007, inachi-006): Hairline crack on concrete surface, appears to be relatively uniform in width along its length, consistent with settlement or shrinkage cracking. NOTE: no crack visible; the horizontal feature is a ledge/edge of a rough white surface (dataset: no_defect).
- [x] photo_4 (eng-053, eng-011, eng-056): Concrete surface showing brown staining concentrated along a vertical crack, with rust-colored discoloration at the crack mouth indicating potential reinforcement corrosion.
- [x] photo_5 (hud-007, inachi-004): Hairline crack visible on concrete surface with minimal visible staining. Crack appears relatively narrow and short.

## walk_02
- [ ] photo_1 (hud-007): Foundation or concrete wall surface with fine hairline cracking visible. ... NOTE: uniform stippled render, no crack visible (dataset: non-cracked) - over-call.
- [ ] photo_2 (inachi-007): Concrete surface with visible cracking and what appears to be white mineral deposits (efflorescence) along crack lines. ... NOTE: the crack is real and clear; the efflorescence claim is not supported (no white deposits visible along it) - half right.
- [ ] photo_3 (eng-064): Interior drywall or wall surface with peeling and flaking paint. ... NOTE: the image is a near-black, featureless crop; nothing described is discernible - misread.
- [x] photo_4 (inachi-042, inachi-046): Interior stair wall with visible peeling paint and surface deterioration. The paint is blistering and flaking, suggesting moisture trapped behind the paint film or a moisture source within the wall assembly rather than simple weathering.
- [x] photo_5 (inachi-033, hud-023): Exterior siding near downspout area showing greenish discoloration and streaking on the surface. The growth pattern is consistent with algae or mold colonization in a shaded, damp area. The location near the downspout suggests potential moisture management issues.

## walk_03
- [x] photo_1 (eng-075, inachi-035): Greenish diffuse staining and discoloration on a light-colored surface, concentrated in one area with streaking pattern characteristic of algae or mold growth associated with chronic moisture.
- [x] photo_2 (inachi-051): Exterior concrete walkway surface showing fine cracking pattern typical of concrete curing shrinkage, with no visible vertical offset, spalling, or active widening.
- [ ] photo_3 (hud-007): Concrete surface with visible fine cracking; cracks appear hairline in width and do not show clear signs of active movement or displacement. NOTE: no crack visible; stippled pavement with a small surface pit (dataset: non-cracked) - over-call.
- [ ] photo_4 (eng-062, inachi-042): Interior stairwell wall showing blistered and peeling paint with visible moisture damage pattern. ... NOTE: the image shows a drawn/graffiti geometric animal figure on stucco, not blistered paint - misread.
- [ ] photo_5 (hud-007): Concrete surface with fine hairline cracking visible; ... NOTE: uniform gray concrete, no crack visible (dataset: non-cracked) - over-call.

## walk_04
- [x] photo_1 (inachi-009, hud-011): Exterior concrete flatwork (likely balcony or patio surface) with scattered shallow spalling and surface flaking visible across the area, consistent with freeze-thaw deterioration. The spalling appears to be minor surface-level loss with a pitted texture.
- [x] photo_2 (eng-057, inachi-052): Interior ceiling with rust-colored streaks and discoloration directly below the drip edge, as noted by the technician. The streaks appear concentrated in a pattern consistent with water drainage or leakage from above. (Blurry image; "streaks" is generous, but the rust-colored discoloration is real.)
- [ ] photo_3 (inachi-007, inachi-006): Concrete surface with visible cracking. The crack appears to show efflorescence staining ... NOTE: the vertical lines are precast formwork grooves/joints, not cracks (dataset: no_defect) - misread.
- [x] photo_4 (inachi-040, hud-022): Interior ceiling or wall surface with water staining and discoloration visible. The staining pattern suggests moisture intrusion from above, potentially linked to roof or attic conditions.
- [x] photo_5 (eng-072): Clean, dry concrete surface with sound consolidation, uniform color, and no visible cracking, rust staining, or efflorescence. The surface appears well-maintained with no active deterioration.

## walk_05
- [x] photo_1 (inachi-007, hud-025): Foundation wall with white powdery deposits and visible discoloration. Technician noted white deposits present. The deposits appear chalky and mineral in nature consistent with efflorescence. (Deposits are subtle but present; note-anchored.)
- [ ] photo_2 (inachi-019, inachi-017, inachi-018): Foundation wall surface showing white mineral deposits in a banded or patchy distribution pattern. ... NOTE: light coarse-aggregate texture, not distinct mineral deposits (dataset: non-cracked pavement) - over-read.
- [ ] photo_3 (hud-007, inachi-003): Concrete surface with a visible diagonal crack. ... NOTE: no clear diagonal crack visible on the uniform stippled surface (dataset: non-cracked) - over-call.
- [x] photo_4 (inachi-003, inachi-004): Area above window showing a diagonal crack pattern. Technician noted this location is above the window. The crack appears to radiate from the window corner area, consistent with stress concentration at the opening.
- [ ] photo_5 (eng-072): Concrete surface appearing sound ... no ... staining observed in this view. NOTE: mostly right, but faint yellowish discoloration IS present; "no staining" overstates - half right.

## walk_06
- [x] photo_1 (inachi-022, inachi-021, hud-008): Underside of slab edge showing exposed reinforcing steel with visible corrosion and rust staining. The rebar appears to have insufficient concrete cover, with the steel directly exposed to the environment.
- [ ] photo_2 (inachi-007, eng-053, eng-023): Concrete surface with white efflorescence deposits and visible cracking. ... NOTE: mottled dirt/staining on pale concrete with only a faint line; "efflorescence concentrated along crack pathways" is an over-read (dataset: no_defect).
- [x] photo_3 (eng-072): Concrete surface showing sound, well-consolidated material with no visible active deterioration, cracking, or rust staining evident in this section.
- [x] photo_4 (inachi-046, hud-029, hud-034): Soffit surface with visible peeling and flaking paint, exposing underlying material. The paint failure appears localized to this area with no visible water staining or mold growth directly visible.
- [x] photo_5 (epa-mold-001, hud-023): Lower wall surface showing visible mold or mold-like growth with green and dark discoloration, appearing at moderate visual extent across a localized area. The growth is accompanied by water staining, indicating moisture intrusion.
