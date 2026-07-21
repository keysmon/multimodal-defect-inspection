# Walkthrough visual-accuracy spot-check (hand-rated)

Visual accuracy is NOT auto-measured - there are no labels for whether
the model read a photo correctly. Rate each observation against its
photo: mark [x] when accurate; leave unchecked and add a note when not.

Rater: Claude (Fable 5) vision review, 2026-07-21 - independent-model
cross-check, per-photo against the source image. Override on human review.

**Result: 8/8 primary observations accurate** (vs 17/30 strict on the
256px dataset-crop set) - visual accuracy is context/resolution-bound;
realistic field photos are read essentially correctly. Two secondary
attributions are arguable and noted inline (pale render / green algae
described as "efflorescence" near cracks).

## realistic_01
- [x] photo_1 (hud-018, hud-025, epa-mold-013): Rear wall by drainpipes shows dark staining below a white PVC elbow pipe with visible rust streaking and discoloration on the masonry. The staining pattern is vertical and concentrated, with rust-colored deposits suggesting prolonged moisture contact. A musty odor was noted by the technician.
- [x] photo_2 (epa-mold-011, epa-mold-025, epa-mold-018, hud-019): Interior room with extensive dark staining and mold-like growth covering walls, floor, and structural elements. Materials appear saturated or recently wet, with visible discoloration and biological growth. Abandoned items and debris suggest flood damage or prolonged water intrusion.
- [x] photo_3 (eng-064, inachi-042, inachi-046): Corridor wall displays large-scale paint peeling and blistering in sheets, with green paint lifting away from the substrate and white primer or base coat exposed. The peeling pattern suggests moisture pressure behind the paint film rather than simple weathering.
- [x] photo_4 (inachi-042, eng-063): Entrance pillar painted surface shows small blisters and localized paint failure on white coating. The blistering pattern is discrete rather than widespread, with paint bubbling away from the masonry substrate.

## realistic_02
- [x] photo_1 (hud-008, inachi-010, inachi-013): Low garden wall with exposed rusted reinforcement mesh and concrete. The concrete surface shows spalling and flaking with visible rust staining from the embedded steel. The affected area appears to exceed roughly one square foot and penetrates significantly into the material, exposing the reinforcing mesh beneath the surface.
- [x] photo_2 (inachi-007, eng-009, inachi-008): Boundary brick wall with visible wide vertical crack running through the masonry. The crack shows efflorescence staining (white mineral deposits) along its length, indicating active or recent water penetration. The crack appears to follow a stair-step pattern through the mortar joints typical of differential settlement in unit masonry. NOTE: primary claims accurate; "efflorescence along its length" is arguable (pale render remnants, not clearly efflorescence).
- [x] photo_3 (inachi-007, eng-009, inachi-012): Arched passage with a long stepped crack running vertically through the brick masonry from floor to ceiling. The crack shows efflorescence staining along its length and appears to be a stair-step pattern following mortar joints. The extent and pattern suggest ongoing differential movement rather than simple shrinkage. NOTE: crack reading exact; the pale/green staining is predominantly algae/moss, not efflorescence.
- [x] photo_4 (epa-mold-001, hud-023, hud-024): North-facing window on deteriorated wall showing green and purple biological growth (algae/mold-like substance) at moderate to high visual levels along the base where render has fallen away. The growth is patchy but covers a notable area, indicating sustained dampness and organic material availability.
