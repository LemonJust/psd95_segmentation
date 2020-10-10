# Synspy: Synaptic image segmentation using numpy, opencl, volspy, and vispy

[Synspy](http://github.com/informatics-isi-edu/volspy) is an
interactive volume segmentation tool. Synspy is being developed to
support research involving 3D fluorescence microscopy of live
Zebrafish synapses.

## Status

Synspy is experimental software that is subject to frequent changes in
direction depending on the needs of the authors and their scientific
collaborators.

## Using Synspy

Synspy has three usage scenarios:

1. A framework for developing batch image analysis tools that
  can segment and measure very large images.
2. A framework for image segmentation tools, where a basic
  interactive volume rendering capability can complement custom
  data-processing tools.
3. A standalone application for interactively segmenting images with
  our current segmentation algorithm.

### Prerequisites

Synspy is developed primarily on Linux with Python 2.7 but also tested
on Mac OSX. It has several requirements:

- [Volspy](http://github.com/informatics-isi-edu/volspy) volume
  rendering framework.
- [Vispy](http://vispy.org) visualization library to access OpenGL
  GPUs.  A recent development version is needed, and it is tested with
  the fork [karlcz/vispy](https://github.com/vispy/vispy) which is
  periodically synchronized with the upstream project.
- [Numpy](http://www.numpy.org) numerical library to process
  N-dimensional data.
- [Numexpr](https://github.com/pydata/numexpr) numerical expression
  evaluator to accelerate some Numpy operations.
- [PyOpenCL](http://mathema.tician.de/software/pyopencl/) to access
  OpenCL parallel computing platforms.
- [Tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) for
  access to OME-TIFF and LSM microscopy file formats.
- [MDAnalysis](http://pypi.python.org/pypi/MDAnalysis/) for basic
  transformation matrix utilities.
- [Python-PCL](https://github.com/strawlab/python-pcl) bindings for
  point-cloud registration algorithms.
  - [Point Cloud Library (PCL)](http://pointclouds.org/) is the 
    actual library. We test with `pcl-devel` on Fedora Linux.
  - PCL is optional and only used by `synspy.analyze.pair` routines.

### Installation

0. Install all third-party prerequisites.
1. Check out the development code from GitHub for Synspy and Volspy.
2. Install with `python setup.py install` for each of Synspy and Volspy.

### Detecting Synapses

1. Obtain a sample 2 channel 3D TIFF image such as:
   http://www.isi.edu/~karlcz/sample-data/zebra-d19-03b-D.ome.tiff.gz
   **Warning: this is a large 886 MB file!**
2. Launch the viewer `synspy-viewer zebra-d19-03b-D.ome.tiff`
3. Interact with the viewer:
  - Press `ESC` when you have had enough.
  - Press `?` to print UI help text to console output.
  - The mouse can be used to control the view or interact with segments.
	1. Mouse-over on segments will display some pop-up diagnostics.
	2. Dragging *button 1* will rotate the image
	3. Dragging *button 2* or *button 3* will enable temporary slicing
      and cropping modes where the vertical axis controls the depth of
      a cutting plane.
	4. Clicking a segment will toggle its manual classification state.
  - The `f` and `F` keys control the minimum *feature* intensity
    threshold for classifying blobs as synapses.
  - The `n` and `N` keys control the maximum *neighborhood* threshold to
    reject blobs with bright background.
  - The `m` and `M` keys control the maximum *mask* threshold to
    reject blobs that are in autofluorescing regions according to the
    red-channel mask.
  - The `z` and `Z` keys control the *zoom* level.
  - The `g` and `G` keys control the *gain* level.	
  - The number keys `0` to `9` with and without shift modifier set
    *gain*.
  - The `t` and `T` keys control the *transparency* floor, the voxel
    intensity at which opacity is 0.0.
  - The `u` and `U` keys control the transparency *upper bound*, the
    voxel intensity at which opacity is 1.0.
  - The `o` and `O` keys control the *opacity* factor which is applied
    after the linear ramp from floor to upper bound.
  - The `b` key cycles through feature blending mode for
    on-screen rendering, where unclassified voxels are rendered in
    blue:
    1. Linear intensity mapping for voxels within classified segments
      (green for synapses, red for auto-fluorescence).
    2. Full intensity fill for voxels within classified segments
      (green for synapses, red for auto-fluorescence).
  - The `d` key *dumps* current parameters to the console output.
  - The `D` key *dumps* a voxel classification TIFF image and a
    segment list CSV file, as well as current threshold parameters.
  - The `l` key *loads* a previously dumped segment list CSV file to
    continue making manual classification decisions in a new
    session. This also loads the threshold parameters saved to a
    special row of the CSV file, if found. (For backwards
    compatibility, older CSV dumps without saved threshold parameters
    are also supported, leaving the thresholds unchanged.)
  - The `h` key writes out a 2D histogram of all blobs using the
    feature intensity and background noise intensity measures as the
    two plotting axes.
  - The `e` key *endorses* currently in-threshold segments as if they
    have been manually classified as true synapses.
  - The `E` key *expunges* manually classified segments, making those
    segments unclickable so that other adjacent segments are easier to
    click.

Do not be alarmed by the copious diagnostic outputs streaming out on
the console. Did we mention this is experimental code?

### Detecting Nuclei

Follow the same procedure as above, except set the environment
variable `SYNSPY_DETECT_NUCLEI=true` when launching the
`synspy-viewer` application. This changes the size of the footprints
used for scale-sensitive blob detection, so that large nuclei scale
blobs are detected and classified instead of small synapse-scale
blobs.

### Registering Two Images

We have only rudimentary support for registering multiple images when tracked
in the catalog:

1. Acquire two images, such as a *before* and *after* image of the same tissue and submit to catalog.
2. Define image regions for nucleic and synaptic regions of interest to compare.
3. Use the `synspy-launcher` GUI to perform segment classification on all four regions.
4. Create an image pair study for the two nucleic regions.
5. Create a synaptic pair study for the two synaptic regions.
6. Wait for the catalog to produce registered pointclouds.
5. Launch `synspy-register syn_study_id` to interactively view the aligned pointclouds with a pairwise matching algorithm:
   - This requires credentials obtained in advance via the `deriva-auth` authentication agent

In the interactive viewer, the `1` ... `0` keys can be pressed to
toggle the visibility of the sub-plots:
- `1`: unmatched nuclei from image 1
- `2`: matched nuclei from image 1
- `3`: segments connecting matched nuclei
- `4`: matched nuclei from image 2
- `5`: unmatched nuclei from image 2
- `6`: unmatched synapses from image 1
- `7`: matched synapses from image 1
- `8`: segments connecting matched synapses
- `9`: matched synapses from image 2
- `0`: unmatched synapses from image 2

The point matching and the pointcloud viz can be adjusted by environment parameters:
- `SYNSPY_HOST`: the hostname of the server (default `synapse.isrd.isi.edu`)
- `SYNSPY_CATALOG`: the catalog identifier (default `1`)
- `DERIVA_CREDENTIALS`: the filename where credentials are stored (default unset to use those from `deriva-auth`)
- Nucleic pairing parameters
   - `NUC_PAIRING_RADIUS`: distance allowed between paired nuclei in microns (default 15.0)
   - `NUC_CORE_DX_RATIO`: intensity units/micron for 4D nearest-neighbor (default unset for 3D nearest-neighbor)
   - `NUC_MAX_RATIO`: maximum intensity ratio to accept for pairs
- Synaptic pairing parameters
   - `SYN_PAIRING_RADIUS`: distance allowed between paired synapses in microns (default 5.0)
   - `SYN_CORE_DX_RATIO`: intensity units/micron for 4D nearest-neighbor (default unset for 3D nearest-neighbor)
   - `SYN_MAX_RATIO`: maximum intensity ratio to accept for pair
- `BACKGROUND_RGB`: 3-channel RGB in normalized 0.0-1.0 space (default `0.15,0.15,0.15` for a dark gray)
- `SHOW_AXES`: `true` enables and `false` disables axes arrows (default `true`)

### Environment Parameters

Several environment variables can be set to modify the behavior of the `synspy-viewer` tool on a run-by-run basis, most of which are in common with the `volspy-viewer`:

- `PEAKS_DIAM_FACTOR` (default `0.75`) controls the diameter of a gaussian low-pass filter that is used to smooth the image before doing local maxima detection. This factor adjusts the diameter relative to the built-in synapse-diameter that is used to model the core intensity distribution of synapse candidates. A smaller diameter will allow the detection of more closely spaced local maxima but may introduce errors as pixel-level noise begins to dominate.
- `DUMP_PREFIX` controls how dumped files are named. When running on `dir1/ImgZfAbc20161231A1.ome.tiff` the default behaves as if you specified `DUMP_PREFIX=./ImgZfAbc20161231A1.` and will name dump files such as `./ImgZfAbc20161231A1.synapses-only.csv`.
- `VOXEL_SAMPLE` selects volume rendering texture sampling modes from `nearest` or `linear` (default for unspecified or unrecognized values).
- `VIEW_MODE` changes the scalar field that is volume-rendered:
  - `raw` renders the raw data (default)
  - `dog` renders a difference-of-gaussians transform to emphasize synapse-scale changes
- `ZYX_SLICE` selects a grid-aligned region of interest to view from the original image grid, e.g. `0:10,100:200,50:800` selects a region of interest where Z<10, 100<=Y<200, and 50<=X<800. (Default slice contains the whole image.)
- `ZYX_VIEW_GRID` changes the desired rendering grid spacing. Set a preferred ZYX micron spacing, e.g. `0.5,0.5,0.5` which the program will try to approximate using integer bin-averaging of source voxels but it will only reduce grid resolution and never increase it. NOTE: Y and X values should be equal to avoid artifacts with current renderer. (Default grid is 0.25, 0.25, 0.25 micron.)
- `ZYX_IMAGE_GRID` overrides any voxel grid spacing information from the image file metadata, useful in case it is absent or incorrect.
- `ZYX_BLOCK_SIZE` changes the desired sub-block work unit size for decomposing large images to control Numpy or OpenCL working set size. Set a preferred ZYX voxel count, e.g. `256,384,512` which the program will try to approximate to find an evenly divisible block layout.
- `MAX_3D_TEXTURE_WIDTH` hints the per-dimension size of the volume cube loaded into an OpenGL texture. This affects the sampling precision along the ray-casting integration in the renderer. (Default is `768`.)
- `ZNOISE_PERCENTILE` enables a sensor noise estimation by calculating the Nth percentile value along the Z axis, e.g. `ZNOISE_PERCENTILE_5` estimates a 2D noise image as the 5th percentile value across the Z stack, and subtracts that noise image from every slice in the stack as a pre-filtering step. *WARNING*: use of this feature causes the entire image to be loaded into RAM, causing a significantly higher minimum RAM size for runs with large input images. (Default is no noise estimate.) 
  - `ZNOISE_ZERO_LEVEL` controls a lower value clamp for the pre-filtered data when percentile filtering is enabled. (Default is `0`.)
- `SYNSPY_AUTO_DUMP_LOAD=true`: Automatically load an existing segments CSV file on startup and dump one on exit via the `ESC` key. See related `DUMP_PREFIX` for how segments CSV file must be named. **WARNING***: exiting the application by other means such as using the `[X]` close button on the window decoration MAY bypass the application and exit without dumping the latest voxel classifications.
- `SYNSPY_SPLAT_SIZE` allows some influence over the size of the splat used to represent centroids in the renderer. The valid range is `0.0` to `2.0` (default `1.0`). The splat is always derived from a threshold on a 3D gaussian distribution in a small box, with at least the central voxel selected. The value `0` is the smallest possible splat.  The value `1` attempts to preserve the legacy spheroid splat. The value `2` expands the splat to completely fill the enclosing box. Other intermediate values will select incrementally larger or smaller voxel sets between those extremes.

The `ZYX_SLICE`, `ZYX_VIEW_GRID`, and `MAX_3D_TEXTURE_WIDTH` parameters have different but inter-related effects on the scope of the volumetric visualization.

1. The `ZYX_VIEW_GRID` can control down-sampling of voxels in arbitrary integer ratios, e.g. to set a preferred grid resolution that can differentiate features of a given size without wasting additional storage space on irrelevant small-scale details. This can save overall RAM required to store the processed volume data by reducing the global image size. The down-sampling occurs incrementally as each sub-block is processed by the block-decomposed processing pipeline.
1. The `ZYX_SLICE` can arbitrarily discard voxels and thus reduce the final volume size, though discarded voxels may be temporarily present in RAM and require additional memory allocation at that time.
1. The `MAX_3D_TEXTURE_WIDTH` can avoid allocating oversized OpenGL 3D textures which would either cause a runtime error or unacceptable performance on a given hardware implementation. This can save overall texture RAM required to store the volume data on the GPU, but actually increases the host RAM requirements since it generates a multi-resolution pyramid on the host from which different 3D texture blocks are retrieved dynamically.

## Help and Contact

Please direct questions and comments to the [project issue
tracker](https://github.com/informatics-isi-edu/synspy/issues) at
GitHub.

## License

Synspy is made available as open source under the (new) BSD
License. Please see the [LICENSE
file](https://github.com/informatics-isi-edu/synspy/blob/master/LICENSE)
for more information.

## About Us

Synspy and Volspy are developed in the [Informatics
group](http://www.isi.edu/research_groups/informatics/home) at the
[USC Information Sciences Institute](http://www.isi.edu).  The
computer science researchers involved are:

* Karl Czajkowski
* Carl Kesselman

