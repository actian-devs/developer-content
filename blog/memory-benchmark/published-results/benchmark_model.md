## Article table

| Vectors | Precision | Raw vectors (N·D·B) | Measured memory | Measured disk |
|---|---|---|---|---|
| 1,000 | Float32 | 1.46 MiB | 39.4 MiB | 4.4 MiB |
| 1,000 | Int8 | 0.37 MiB | 29.9 MiB | 4.9 MiB |
| 10,000 | Float32 | 14.65 MiB | 133.8 MiB | 60.0 MiB |
| 10,000 | Int8 | 3.66 MiB | 83.4 MiB | 24.8 MiB |
| 100,000 | Float32 | 146.48 MiB | 748.9 MiB | 488.5 MiB |
| 100,000 | Int8 | 36.62 MiB | 345.9 MiB | 435.8 MiB |

## Model fit: growth(N) = fixed + slope·N  (dim=384, memory source: cgroup anon)

| Precision | Fixed cost (fit, MiB) | Empty collection (MiB) | Memory B/vector | Disk B/vector | Raw B/vector (D·B) | Memory ÷ raw | R² |
|---|---|---|---|---|---|---|---|
| Float32 | 47.2 | 15.1 | 7,374 | 5,073 | 1,536 | 4.8x | 0.998 |
| Int8 | 39.1 | 15.2 | 3,230 | 4,655 | 384 | 8.4x | 0.994 |

## Per checkpoint

| Vectors | Precision | Measured (MiB) | Measured − fixed (MiB) |
|---|---|---|---|
| 1,000 | Float32 | 39.4 | -7.81 |
| 10,000 | Float32 | 133.8 | 86.65 |
| 100,000 | Float32 | 748.9 | 701.72 |
| 1,000 | Int8 | 29.9 | -9.20 |
| 10,000 | Int8 | 83.4 | 44.32 |
| 100,000 | Int8 | 345.9 | 306.83 |
