# Pre-processing workflows

This guide gives the recommended order of steps for each platform. Steps in *italics* are optional or depend on the data.
Each step names the `hyperprep` function and, in brackets, the matching pipeline step.

---

## 1. Terrestrial scanners (tripod / outcrop / mine face / core and lab scanners)

### 1a. Lab and core scanners (controlled illumination)

1. Sensor corrections, which convert raw DN to reflectance using the white and dark references:
   `calibration.sensors.correct_sensor_folder` (`sensor_correction`). This step uses hylite and covers FENIX, FX10/17/50,
   OWL, Rikola and Telops. For other sensors, use `calibration.white_reference_reflectance` (`white_reference`) on ENVI files.
   * Set `white_panel` to the panel used in the session (for example `R90`, or `R55_SiSu` for a grey panel).
   * FENIX: `flip=True` if the camera was mounted backwards, and `shift=True` for the VNIR-SWIR spatial offset (near-field only).
2. *Bad detector pixel repair*: `noise.repair_detector_pixels` (`repair_bad_pixels`), when the sensor class didn't already do `bpr`.
3. *Destriping*: `noise.destripe` (`destripe`).
4. Bad-band removal: `spectral.remove_bad_bands` (`remove_bad_bands`).
5. Smoothing: `spectral.savgol` (`savgol`), and *spike removal* with `noise.remove_spikes`.
6. Value-range check: `spectral.clip_range` (`clip_range`).

### 1b. Outdoor scenes (natural illumination, 3-D geometry)

1. Sensor corrections to radiance (as in 1a, with `rad=True` and without reflectance conversion).
2. Projection onto a point cloud:
   * frame or panoramic cameras whose pose is stored in the header: `terrestrial.projection.project_frame_scene` or `build_scene`;
   * pushbroom cameras with an IMU track: `drone.read_hyspexnav` → `terrestrial.projection.build_pushbroom` →
     *`crop_track`* → *`solve_boresight`* → `build_scene`.
3. Illumination / atmospheric correction, using one of two approaches:
   * **Illumination model** (one scene, with a sunlit panel): `terrestrial.illumination.correct_scene`. This computes the sun
     vector from the time and location, Oren-Nayar direct illumination factors from the cloud normals, and sky-view
     factors, solves for the sun and sky spectra, and inverts `R = L / (skv·S + ilf·I)`.
   * **ELC + topographic correction** (several scenes): `terrestrial.hypercloud.correct_scene`. This applies an ELC from
     the panels stored in the header, then `topo.topographic_correction` (`ambient`, `cfac`/`c`, `cosine`, `minnaert`)
     using the scene normals and the header sun angles.
4. *Brightness filtering*: `noise.percentile_brightness_filter` removes uncorrected shadows and false highlights.
5. *Colour matching between scenes*: `terrestrial.hypercloud.colour_match`.
6. Blending into a hypercloud: `terrestrial.hypercloud.blend` (`average`, `gsd`, `distance`, `obliquity`).
7. Export: `terrestrial.hypercloud.export_hypercloud` (hypercloud PLY plus colourised false-colour clouds).
8. Spectral clean-up on the cloud or on rendered images: `savgol`, `hull_correct`, `spatial_smooth`.

`terrestrial.hypercloud.build_hypercloud` runs steps 2 to 7 for a folder of images.

---

## 2. Drone / airborne pushbroom

1. Read radiance (`read`). For HySpex RAD integer output, decompress with `calibration.apply_scale_factor` (`hyspex_scale`).
   If you have raw DN, use `calibration.dn_to_radiance` (`dn_to_radiance`) with the sensor's gain and offset.
2. No-data masking: `masks.mask_nodata` (`mask_nodata`).
3. Sensor artefacts: `noise.repair_detector_pixels` (`repair_bad_pixels`), then `noise.destripe(method='ratio', window=31)`
   (`destripe`). The window keeps real cross-track gradients. Optionally, use `remove_spikes`.
4. Reflectance, using one of:
   * **ELC** with targets in the flight line: `calibration.Panel.from_region` + `calibration.empirical_line` (`elc`).
     Use two or more panels to also remove path radiance.
   * **Downwelling irradiance sensor**: `drone.irradiance_normalisation` (`irradiance_normalisation`), `R = π L / E`.
   * Then *`drone.sun_angle_normalisation`* when flights span several hours.
5. Cross-track BRDF / vignetting: `drone.cross_track_correction` (`cross_track_correction`). Fit it on a homogeneous cover
   class with `mask` if the scene is very heterogeneous.
6. Bad-band removal and smoothing (`remove_bad_bands`, `savgol`).
7. Geometry, using one of:
   * **2.5-D terrain**: `drone.read_hyspexnav` / `read_nav_csv` → *`NavTrack.apply_lever_arm`* → `drone.orthorectify`
     with a `DEM` or a constant ground height, and the boresight angles (`orthorectify`). The result is a north-up map
     image and a GLT that you can reuse for other layers with `ortho.apply_glt`.
   * **Steep 3-D terrain**: point-cloud projection, as in 1b step 2.
8. Flight-line matching and mosaicking: `drone.match_flightlines` (relative normalisation in overlaps), then
   `ortho.mosaic` (`first`, `mean`, `weighted`, or `best` with view-zenith weights from `orthorectify(..., return_glt=True)`).

Attitude convention: roll, pitch and yaw are in degrees, body frame forward-right-down, NED navigation frame, yaw is the
heading. This matches PosPac and HySpexNav exports. Positive roll (right wing down) makes a nadir sensor look to the left.

---

## 3. Satellite (EnMAP, PRISMA, EMIT, DESIS, others)

### L1 radiance

1. Read: `satellite.read_product` (`read`, with `sensor:` to force the reader).
   * EMIT: `ortho=True` applies the product GLT. Otherwise, keep raw geometry for smile and stripe work and apply the GLT
     afterwards with `satellite.emit_apply_glt`.
   * PRISMA: latitude and longitude layers are in `meta['lat']` / `meta['lon']`. Convert them with `geo.latlon_to_utm`
     and use `ortho.build_glt` to orthorectify.
2. VNIR/SWIR overlap: `satellite.merge_overlap` (`merge_overlap`).
3. In raw geometry: *smile* with `satellite.estimate_smile` (O2-A band) → `satellite.smile_correction` (`smile_correction`),
   and *striping* with `noise.destripe` (`destripe`).
4. TOA reflectance: `satellite.toa_reflectance` (`toa_reflectance`). Check `radiance_scale` (EMIT: 0.01;
   PRISMA L1: 0.001) and supply a solar spectrum with `esun_path`.
5. Surface reflectance (scene-based), using one of: `satellite.dark_object_subtraction` (`dos`); ELC with field spectra
   (`elc`, panels from `radiance_file` or a `region`); or `calibration.iarr` / `flat_field`. For physics-based correction,
   use L2A products or ISOFIT.
6. Masks: `masks.cloud_mask` (+ `dilate`), `shadow_mask`, `water_mask` (`mask_clouds`, `mask_shadows`, `mask_water`).
7. *Topographic correction* with a co-registered DEM: `topographic_correction` (pipeline) or `topo.*`.
8. Bad bands, smoothing, *MNF denoising* (`remove_bad_bands`, `savgol`, `mnf_denoise`).
9. *Subset* to an area of interest: `satellite.subset_bbox` (`subset_bbox`). *Resample* to another sensor:
   `spectral.resample_image` (`resample` with `like:`).

### L2 surface reflectance

`read` → `reflectance_scale_check` → masks → `remove_bad_bands` → `savgol` → *`subset_bbox`* / *`resample`*.

---

## Where each notebook cell went

| Notebook | Cell(s) | Replacement |
|---|---|---|
| `1_radiance_to_reflectance_Fenixsingle` | white panel choice, `Fenix/FX50/OWL.correct_folder` | `calibration.sensors.correct_sensor_folder(path, sensor, white_panel, **kwds)` |
| | save with `saveWithGDAL` | `outdir=` argument, or `hyperprep.save` |
| `2_Projection_mapping` | HySpex scale factor, zero masking | `calibration.hyspex_scale_factor` / `apply_scale_factor` |
| | IMU text → UTM (zone hard-coded as 33S) | `drone.read_hyspexnav` (zone detected automatically, no `utm` dependency) |
| | line count check | `NavTrack.check_lines`, `build_pushbroom(drop_extra_lines=True)` |
| | `Pushbroom(...)`, `crop`, `optimize_boresight` | `terrestrial.projection.build_pushbroom`, `crop_track`, `solve_boresight` |
| | `HyScene.construct` | `terrestrial.projection.build_scene` |
| | three duplicated false-colour cells | `terrestrial.projection.false_colour_cloud` |
| `3_Illumination_corrections_FENIX` | panel text files | `calibration.Panel.from_file`, `io.read_spectrum_txt` |
| | sun vector from header (time overridden by hand) | `terrestrial.illumination.scene_sun_vector` / `solar.sun_vector` |
| | panel alpha (normal negated by hand) | `terrestrial.illumination.panel_illumination_factor` |
| | `calcOrenNayar`, sky view | `topo.oren_nayar`, `illumination_factors`, `topo.skyview_from_normals` |
| | `estimate_illu`, `IlluModel.getReflectance` | `estimate_sky_sun` (closed form) or `method='hylite'`; `illumination_reflectance` |
| | `get_hull_corrected`, `blur`, `smooth_savgol` | `spectral.hull_correct`, `spectral.spatial_smooth`, `spectral.savgol` |
| `1b_Build_and_correct_hypercloud_ambient` | image collection, bad-band windows | `hypercloud.collect_images`, `common_wavelengths`, `spectral.HYPERCLOUD_BAD_WINDOWS` |
| | `HyScene(image, cloud, cam)` (removed from hylite) | `projection.project_frame_scene` |
| | `scene.correct(...)` (removed from hylite) | `hypercloud.correct_scene` (ELC + `topo.topographic_correction`) |
| | reflectance range masking | `spectral.clip_range(action='nan_pixel')` |
| | `match_colour_to` loop | `hypercloud.colour_match` |
| | `HyScene.build_hypercloud` (removed from hylite) | `hypercloud.blend` (`get_blend_weights` + `blend_scenes`) |
| | RGB / SWIR exports, compression | `hypercloud.export_hypercloud` |

### Bugs fixed while porting

* Notebook 1 called `Fenix.set_white_ref_spectra` without importing `Fenix`, and used both `image` and `images`.
* Notebook 1b deleted the bad-band list from the wrong image variable, and its failure message printed the wrong path.
* Notebook 3 unpacked `estimate_illu` as `(skylight, sunlight)`, but current hylite returns `(sun, sky)`.
* Notebook 3 replaced the header time with a hard-coded value after printing the header time.
* Notebook 1b used `HyScene` and `build_hypercloud` APIs that no longer exist in current hylite (1.4x).
