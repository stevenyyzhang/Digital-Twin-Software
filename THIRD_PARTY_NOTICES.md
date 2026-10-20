# Third-party notices

## Software

This application derives from NVIDIA's `sionna-rt-gui`:
https://github.com/NVlabs/sionna-rt-gui

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Upstream code is distributed under Apache-2.0; the original license text and
source headers are retained. Local changes include the two-workspace UI,
beam selection, asynchronous coverage, layered maps, scene presentation,
and responsive display rendering. Upstream authorship is not a claim that
NVIDIA developed or endorsed those modifications.

Dependencies retain their respective licenses. They are installed separately,
not vendored into this repository. The software license does not relicense
third-party models, imagery, or databases.

The Windows portable release additionally bundles Python and the installed
dependencies. Python's license is at `runtime/LICENSE.txt`; dependency license
files are retained in their package directories and `runtime/Lib/site-packages`
metadata directories (including `.dist-info/licenses` where supplied).
These licenses remain applicable independently of this project's software license.

## Hong Kong scene data

© The Government of the Hong Kong Special Administrative Region.
Geometry and district terrain imagery: Lands Department, 3D Visualisation Map,
https://3d.map.gov.hk/

Source: CSDI Portal / Open3Dhk, 3D Visualisation Map (Individualised models).
Dataset metadata and directly inspected download terms:
https://portal.csdi.gov.hk/csdi-webpage/metadata/landsd_rcd_1671676915450_88604/html
https://3d.map.gov.hk/download-api
https://portal.csdi.gov.hk/csdi-webpage/doc/TNC

Original intellectual property remains with the Government and relevant
organizations. Project processing includes coordinate conversion, cropping,
simplification, material grouping, texture conversion, recoloring, window
styling and display normal processing.
These are modified project assets, not unmodified official releases.
Government endorsement is not implied. Redistribution remains subject to the
source terms; the project does not replace those terms with Apache-2.0.

Preserve source links, ownership declarations and modification notices when
redistributing these assets. See the per-scene notices for the applicable scope:
[Campus](polyu-stylized/ASSET_LICENSE.md),
[Mong Kok](mong-kok/ASSET_LICENSE.md),
[Kam Tin](kam-tin/ASSET_LICENSE.md).

## OpenStreetMap-derived campus surfaces

Map data © OpenStreetMap contributors:
https://www.openstreetmap.org/copyright

Database license: Open Database License (ODbL) 1.0:
https://opendatacommons.org/licenses/odbl/1-0/

The campus auxiliary ground and roads include OSM-derived geometry. These
data and relevant derived-database obligations remain separate from the
software's Apache-2.0 license. The original extract and the complete derived
support mesh layers are supplied under ODbL; see the file-level scope in
[the campus notice](polyu-stylized/ASSET_LICENSE.md).

## Environment image

`sionna-rt-gui/src/sionna_rt_gui/data/envmaps/teufelsberg_ground_2_1k.exr`
comes from Poly Haven:
https://polyhaven.com/a/teufelsberg_ground_2

It is retained from upstream with its source attribution. Poly Haven's
[asset license](https://polyhaven.com/license) confirms CC0 redistribution.
The default customized demo uses a generated studio environment instead.
