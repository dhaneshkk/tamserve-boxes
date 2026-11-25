//! Site file. A superset of iogp-editor's ingest `Site` JSON: an existing site file loads unchanged
//! (cameras, px_per_metre, image-space zones), and every new field is optional.
//!
//! ```json
//! { "cameras": [ {
//!     "name": "CAM-02", "image_size": [1280, 720],
//!     "lens": {"K": [[...3],[...3],[...3]], "D": [k1,k2,p1,p2,k3]},  // optional
//!     "projection": [[...4], [...4], [...4]],          // optional 3x4 P (deck frame, metres, Z up)
//!     "decks": [ { "id": "main", "elevation_m": 0.0,
//!                  "image_polygon": [[x,y], ...],      // where in the image this deck is visible
//!                  "homography": [[...3],[...3],[...3]] } ],   // image -> deck metres; omit to derive from P
//!     "occluders": [ [[x,y], ...] ],                   // equipment that hides feet (image polygons)
//!     "zones": [ { "id": "Crane-Deck", "type": "lifting-zone", "polygon": [[x,y],...],
//!                  "deck": "main", "floor_polygon": [[X,Y], ...] } ] } ] }
//! ```

use crate::geom::{plane_homography, Lens, M3};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, Default)]
pub struct Site { #[serde(default)] pub cameras: Vec<SiteCamera> }

#[derive(Clone, Debug, Deserialize, Default)]
pub struct SiteCamera {
    pub name: String,
    #[serde(default)] pub px_per_metre: Option<f64>,
    #[serde(default)] pub image_size: Option<[f64; 2]>,
    /// The lens. Deck homographies and the projection then work on undistorted pixels
    /// (`floor = H · G · undistort(p)`, as tamserve-boxes' `lens`); polygons stay in raw pixels.
    #[serde(default)] pub lens: Option<Lens>,
    #[serde(default)] pub projection: Option<[[f64; 4]; 3]>,
    #[serde(default)] pub decks: Vec<Deck>,
    #[serde(default)] pub occluders: Vec<Vec<[f64; 2]>>,
    #[serde(default)] pub zones: Vec<SiteZone>,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct Deck {
    pub id: String,
    #[serde(default)] pub elevation_m: f64,
    pub image_polygon: Vec<[f64; 2]>,
    #[serde(default)] pub homography: Option<M3>,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct SiteZone {
    pub id: String,
    #[serde(rename = "type")] pub zone: String,
    #[serde(default)] pub polygon: Vec<[f64; 2]>,
    #[serde(default)] pub deck: Option<String>,
    #[serde(default)] pub floor_polygon: Option<Vec<[f64; 2]>>,
}

#[derive(Debug)]
pub struct SiteError(pub String);
impl std::fmt::Display for SiteError { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) } }
impl std::error::Error for SiteError {}

/// A camera ready for per-frame use: deck homographies resolved, zone decks resolved to indices.
#[derive(Clone, Debug)]
pub struct Camera {
    pub name: String,
    pub px_per_metre: Option<f64>,
    pub image_size: Option<[f64; 2]>,
    pub lens: Option<Lens>,
    pub projection: Option<[[f64; 4]; 3]>,
    pub decks: Vec<ResolvedDeck>,
    pub occluders: Vec<Vec<[f64; 2]>>,
    pub zones: Vec<ResolvedZone>,
}
#[derive(Clone, Debug)]
pub struct ResolvedDeck { pub id: String, pub elevation_m: f64, pub image_polygon: Vec<[f64; 2]>, pub h: M3 }
#[derive(Clone, Debug)]
pub struct ResolvedZone { pub id: String, pub zone: String, pub polygon: Vec<[f64; 2]>, pub floor: Option<(usize, Vec<[f64; 2]>)> }

impl Site {
    pub fn from_json(s: &str) -> Result<Site, SiteError> { serde_json::from_str(s).map_err(|e| SiteError(format!("site file: {e}"))) }

    pub fn camera(&self, name: &str) -> Result<Camera, SiteError> {
        let c = self.cameras.iter().find(|c| c.name == name).ok_or_else(|| SiteError(format!("camera {name} not in site file")))?;
        let mut decks = vec![];
        for d in &c.decks {
            let h = match (&d.homography, &c.projection) {
                (Some(h), _) => *h,
                (None, Some(p)) => plane_homography(p, d.elevation_m).ok_or_else(|| SiteError(format!("{name}/{}: projection is degenerate for this deck", d.id)))?,
                (None, None) => return Err(SiteError(format!("{name}/{}: deck needs a homography or a camera projection", d.id))),
            };
            if d.image_polygon.len() < 3 { return Err(SiteError(format!("{name}/{}: image_polygon needs at least 3 points", d.id))); }
            decks.push(ResolvedDeck { id: d.id.clone(), elevation_m: d.elevation_m, image_polygon: d.image_polygon.clone(), h });
        }
        let mut zones = vec![];
        for z in &c.zones {
            let floor = match (&z.deck, &z.floor_polygon) {
                (Some(dk), Some(fp)) => {
                    let i = decks.iter().position(|d| &d.id == dk).ok_or_else(|| SiteError(format!("{name}/{}: unknown deck {dk}", z.id)))?;
                    if fp.len() < 3 { return Err(SiteError(format!("{name}/{}: floor_polygon needs at least 3 points", z.id))); }
                    Some((i, fp.clone()))
                }
                (None, None) => None,
                _ => return Err(SiteError(format!("{name}/{}: set both deck and floor_polygon, or neither", z.id))),
            };
            if floor.is_none() && z.polygon.len() < 3 { return Err(SiteError(format!("{name}/{}: zone has neither an image polygon nor a floor polygon", z.id))); }
            zones.push(ResolvedZone { id: z.id.clone(), zone: z.zone.clone(), polygon: z.polygon.clone(), floor });
        }
        Ok(Camera { name: c.name.clone(), px_per_metre: c.px_per_metre, image_size: c.image_size, lens: c.lens, projection: c.projection, decks, occluders: c.occluders.clone(), zones })
    }
}
