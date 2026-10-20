# Campus scene data notice

Government building geometry: © The Government of the Hong Kong Special
Administrative Region, Lands Department. Source: CSDI Portal / Open3Dhk,
[3D Visualisation Map (Individualised models)](https://portal.csdi.gov.hk/csdi-webpage/dataset/landsd_rcd_1671676915450_88604).
See [download terms](https://3d.map.gov.hk/download-api) and
[CSDI terms](https://portal.csdi.gov.hk/csdi-webpage/doc/TNC).
Original ownership is retained. Local cropping, simplification, recoloring,
window styling and normal processing are not official survey products.
No government or university endorsement is implied.

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
The following OSM-derived data are supplied under
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/):

- `provenance/source.osm`: original local extract, downloaded 2026-10-12.
- `meshes/radio-plaza.ply`, `meshes/real-soft-plaza.ply`
- `meshes/radio-road.ply`, `meshes/real-soft-road.ply`
- `meshes/radio-lawn.ply`, `meshes/real-soft-lawn.ply`
- `meshes/radio-water.ply`, `meshes/real-soft-water.ply`

These complete derivative mesh layers are included in machine-readable PLY
form. Modifications include local coordinate conversion, extrusion with
assumed widths/heights, triangulation, clipping and display normal processing.
The display lawn also includes decorative geometry at OSM tree locations.
The original extract is retained to preserve provenance, not presented as an
unmodified copy of the resulting derived layers.

The two `*-ground.ply` files are schematic rectangular support geometry.
Building material groups (brick, brick_light, roof, white, glass) use government
building geometry; OSM landmark location also informed styling selection.
Do not relicense government geometry under ODbL or Apache-2.0.
Application code licensing is separate from scene-data terms.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for source terms and
the separation between software and scene-data licensing.
