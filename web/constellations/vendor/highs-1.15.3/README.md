# HiGHS WebAssembly dependency

Vendored without modification from the npm package [`highs` 1.15.3](https://registry.npmjs.org/highs/-/highs-1.15.3.tgz), maintained in [lovasoa/highs-js](https://github.com/lovasoa/highs-js).

The wrapper package version is **1.15.3**. The compiled solver reports **HiGHS 1.15.1**, git hash `04024d7`; these versions are intentionally recorded separately.

Only `build/highs.mjs`, `build/highs.wasm`, and the package's MIT [LICENSE](LICENSE) were copied, with the two build files placed in this directory. The workbench loads the JavaScript module and its adjacent WASM file locally; no CDN or external solve service is used at runtime.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `highs.mjs` | 168088 | `f2bdfd071d19c3745756a087026589e9240f329e38315a86f626823ae78a4036` |
| `highs.wasm` | 3531385 | `528be4365bea1d4188988646b244263f50320df55782e14d6af5bce7cd45c840` |
| `LICENSE` | 1090 | `9f8d8dcf27789ac59c0d7c732d6682f148804a5e99d83e9f207d6c1882b98bfa` |

The model, constraints, acceptance checks, and guarantee scope are documented in [the boundary method](../../../../design/zodiac_minimum_corners.md).
