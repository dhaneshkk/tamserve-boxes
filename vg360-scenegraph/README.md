# vg360-scenegraph

This crate turns tracked detections (YOLOX-Nano + ByteTrack from tamserve-boxes) into the objects and facts that
**iogp-runtime** evaluates. Positions are computed on calibrated deck planes in metres instead of
in pixels. It writes through iogp-runtime's zero-copy builder (`begin` / `object` / `fact`), so
the caller just calls `Engine::end()`. Use one `SceneBuilder` per camera, like the engine.

```rust
let mut engine = iogp_runtime::Engine::from_config(&config_json)?;
let site = vg360_scenegraph::Site::from_json(&site_json)?;
let mut sg = SceneBuilder::new(site.camera("CAM-02")?, Params::default(), Policy::from_rules(&engine.rules));
sg.step(t, &dets, camera_guard_ok, &mut engine);
for e in engine.end() { /* alerts */ }
```

## In tamserve-boxes

`src/iogp.rs` runs one `SceneBuilder` + `iogp_runtime::Engine` per camera with `iogp` in its config, using the camera's safety calibration when it has one (see the top-level README, "IOGP Life-Saving Rule alerts").

## Matching iogp-runtime (test1_logpeditor @ 499cf14)

| | |
|---|---|
| object ids | `"<camera>/<track_id>"`, the same as `iogp-editor` ingest |
| class types | detector names (`person`, `hardhat`, `suspended_load`, ...) from `vg360_catalog.json` |
| zones | `Kind::Zone`, id = zone instance name, camera set |
| relations | `in`, `near`, `under`, `on`, `wear`, `carry`, `hold`, `use`, `drive`, `have` (catalog prepositions/verbs; renamable via `RelNames`) |
| `near` distance | taken from the rules' `metres` (if rules disagree, the largest is used and a warning is reported) |
| site file | a superset of ingest's: an existing site file loads unchanged and keeps ingest's pixel semantics |

## What it computes

| relation | with decks/projection | without (ingest parity) |
|---|---|---|
| `in` | the Kalman floor position against the zone's `floor_polygon`, three-valued | foot point in the image polygon |
| `near` | metric distance on the same deck, σ along the pair direction; sort-and-sweep pruning | n/a (deck needed) |
| `under` | drop zone: the load's bottom-centre ray, projected to the deck over `load_height_m`, as a capsule | `below_footprint` |
| `on`, `carry/hold/use`, `drive` | ingest image geometry | same |
| `wear` | ≥ 60 % containment **plus** a vertical band per PPE class (helmet in the top 30 %, ...) | containment only for unbanded classes |

Other behaviour:
* With a `lens` (K, D) in the site camera, points are undistorted before any homography or ray
  (`floor = H · G · undistort(p)`, the same model as tamserve-boxes' `lens.rs`). Image polygons
  (decks, zones, occluders) stay in raw pixels. `SceneBuilder::set_view_correction(G)` takes the
  drift check's correction every frame.
* A fitted homography's sign is arbitrary: negative `w` is accepted, and the deck's image polygon
  decides which pixels belong to it.
* Pixel noise is pushed through the homography Jacobian (numerically when there is a lens), so far-field workers get wider uncertainty.
* When the feet are hidden (by an occluder polygon or the image bottom edge), the foot point is
  recovered from the box top and the person aspect ratio, with inflated σ.
* An occluded person is coasted for up to `coast_max_s`. While coasting, the person keeps asserting what they last wore or
  carried, so an occlusion doesn't become "does not wear a harness".
* `camera_ok = false` (from camera_guard) suppresses every position fact and emits a
  `calibration-lost` signal on each zone. **Add `calibration-lost` to the catalog's `signals`**
  so a rule can alert on it (for example, "If a height-zone has a calibration-lost then ...").

Uncertain relations are emitted with the polarity the rules need: *possibly* if the relation is used positively,
*definitely* if it is used only under negation. `formal/SceneFacts.lean` proves that, given the k·σ error bound, this never
drops a true alert. It also exhibits the one gap: a relation used both positively and negated, which
`Policy` warns about.

## Numbers (release build, this container, 2 cores)

| detections | scene graph | + iogp-runtime (15 fixture rules) |
|---|---|---|
| 6 | 3.2 µs | 5.6 µs |
| 21 | 10.6 µs | 35.0 µs |
| 51 | 39.1 µs | 134.1 µs |

`tests/no_alloc.rs`: 500 steady-state frames with 31 detections and a load cause 0 heap allocations.

## Evaluation

`eval::Confusion` gives per-relation MCC against labelled facts. Report MCC per relation and per
event, not per frame.
