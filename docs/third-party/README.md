# Third-party provenance

The simulation uses third-party noise, rendering, and platform interfaces. These components are not original project-authored implementations.

| Component | Location / provenance | Notice |
| --- | --- | --- |
| FastNoiseLite | `include/FastNoiseLite.h`; Jordan Peck and contributors | Embedded MIT copyright/license retained unchanged. |
| Apple metal-cpp | `external/metal-cpp-main/`; upstream https://developer.apple.com/metal/cpp/ | Apache-2.0 license in vendored `LICENSE.txt`; retain all upstream copyright headers. The vendored README lists macOS 27/iOS 27 first; no exact archive checksum/revision was recorded in this repository. |
| SFML | External SFML 3 dependency, https://www.sfml-dev.org/ | Not vendored; see upstream license with the installed/distributed library. |
| SimplexNoiseFilter | Adapted `src/metal/SimplexNoiseCompute.metal`; Joshua Sullivan | MIT, copyright 2020 Joshua Sullivan; exact pinned upstream license retained in `SimplexNoiseFilter-LICENSE.txt`. |

Simplex source and license were identified at commit `091607969586d33f491fe29fca91802ef9ea50db`:

- https://github.com/JoshuaSullivan/SimplexNoiseFilter/blob/091607969586d33f491fe29fca91802ef9ea50db/Sources/SimplexNoiseCompute/kernel/SimplexNoiseCompute.metal
- https://github.com/JoshuaSullivan/SimplexNoiseFilter/blob/091607969586d33f491fe29fca91802ef9ea50db/LICENSE

## Bundled font: Roboto Mono

The current UI uses an unmodified Roboto Mono font from a pinned Google Fonts revision. Copyright 2015 The Roboto Mono Project Authors. Its SIL Open Font License 1.1 permits bundling and redistribution with software subject to the retained notice and license conditions.

- [Font license](../../assets/fonts/OFL-RobotoMono.txt)
- [Pinned source and SHA-256](../../assets/fonts/README.md)


No project-level license has been chosen. Upstream notices apply to their respective components.
