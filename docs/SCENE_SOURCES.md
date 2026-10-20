# Scene sources and interpretation

| Directory | Scene | Origin |
|---|---|---|
| `polyu-stylized` | PolyU campus | Lands Department buildings; auxiliary OSM-derived ground/roads |
| `mong-kok` | Mong Kok urban district | Lands Department buildings, terrain and infrastructure |
| `kam-tin` | Kam Tin village | Lands Department buildings, terrain and infrastructure |

Each scene includes `scene.xml` (RF geometry/materials), `visuals.json`
(display-only assets), `scenario.json` (illustrative placements/routes),
and the existing build/calibration records where available.

The retained build records specify a 400m crop for Mong Kok and a 500m crop
for Kam Tin. The campus
coverage domain is configured independently at 420m × 535m. Coverage extent
and visible scene bounds are not interchangeable.

Colors, window bands and lighting are presentation choices. A building's color
does not identify its electromagnetic material. The source models are not a
guarantee of current real-world building completeness or surveyed RF behavior.
District Z coordinates retain the model elevations; canopy visible in imagery
does not provide a volumetric vegetation propagation model.

Local edits may simplify reflection surfaces. Materials and transmitter
placements are illustrative; the package contains no calibrated site survey.
See each scene's `ASSET_LICENSE.md` and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for ownership, source terms
and modification notices.
