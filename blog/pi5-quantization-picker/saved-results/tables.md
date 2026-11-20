## The stack before any LLM loads

Board: Raspberry Pi 5 Model B Rev 1.1, 7.87 GiB RAM, swap 2.0 GiB. Retrieval: right chunk in the top 3 for 16/16 answerable questions.

| Part | GiB |
| --- | --- |
| OS and everything else | 0.89 |
| VectorAI DB (4,000-chunk catalog) | 0.15 |
| Ollama with nomic-embed-text | 0.26 |
| Total in use | 1.30 |
| Left for the LLM | 6.57 |

## What fits

| Model | Level | File size (GiB) | Peak RAM used, whole board (GiB) | Swap growth (GiB) | Result |
| --- | --- | --- | --- | --- | --- |
| Phi-4 Mini | Q3_K_M | 1.97 | 4.41 | 0.05 | fits |
| Phi-4 Mini | Q4_0 | 2.17 | 5.15 | 0.14 | fits |
| Phi-4 Mini | Q4_K_M | 2.32 | 5.44 | 0.17 | fits |
| Phi-4 Mini | Q5_K_M | 2.65 | 5.66 | 0.24 | fits |
| Phi-4 Mini | Q6_K | 2.94 | 5.93 | 0.25 | fits |
| Phi-4 Mini | Q8_0 | 3.80 | 6.68 | 0.71 | swaps |
| Gemma 4 E2B | Q3_K_M | 2.36 | 2.25 | 0.00 | fits |
| Gemma 4 E2B | Q4_0 | 2.83 | 3.06 | 0.00 | fits |
| Gemma 4 E2B | Q4_K_M | 2.89 | 3.19 | 0.00 | fits |
| Gemma 4 E2B | Q5_K_M | 3.13 | 3.42 | 0.00 | fits |
| Gemma 4 E2B | Q6_K | 4.19 | 3.79 | 0.01 | fits |
| Gemma 4 E2B | Q8_0 | 4.70 | 4.41 | 0.02 | fits |
| Gemma 4 E4B | Q3_K_M | 3.78 | 3.41 | 0.00 | fits |
| Gemma 4 E4B | Q4_0 | 4.50 | 4.74 | 0.03 | fits |
| Gemma 4 E4B | Q4_K_M | 4.63 | 4.88 | 0.07 | fits |
| Gemma 4 E4B | Q5_K_M | 5.11 | 5.34 | 0.12 | fits |
| Gemma 4 E4B | Q6_K | 6.59 | 5.91 | 1.56 | swaps |
| Gemma 4 E4B | Q8_0 | 7.63 | 6.88 | 1.54 | swaps |
| Qwen3-4B | Q3_K_M | 1.93 | 4.20 | 0.00 | fits |
| Qwen3-4B | Q4_0 | 2.21 | 5.30 | 0.01 | fits |
| Qwen3-4B | Q4_K_M | 2.33 | 5.54 | 0.12 | fits |
| Qwen3-4B | Q5_K_M | 2.69 | 5.72 | 0.12 | fits |
| Qwen3-4B | Q6_K | 3.08 | 6.13 | 0.22 | fits |
| Qwen3-4B | Q8_0 | 3.99 | 6.88 | 1.69 | swaps |

## What holds up

| Model | Level | RAG (of 20) | JSON (of 10) | Instructions (of 10) | Overall |
| --- | --- | --- | --- | --- | --- |
| Phi-4 Mini | Q3_K_M | 20 | 10 | 7 | 92% |
| Phi-4 Mini | Q4_0 | 20 | 10 | 7 | 92% |
| Phi-4 Mini | Q4_K_M | 20 | 8 | 7 | 88% |
| Phi-4 Mini | Q5_K_M | 20 | 9 | 6 | 88% |
| Phi-4 Mini | Q6_K | 20 | 8 | 7 | 88% |
| Phi-4 Mini | Q8_0 | 20 | 7 | 7 | 85% |
| Gemma 4 E2B | Q3_K_M | 20 | 10 | 8 | 95% |
| Gemma 4 E2B | Q4_0 | 20 | 10 | 7 | 92% |
| Gemma 4 E2B | Q4_K_M | 20 | 10 | 9 | 98% |
| Gemma 4 E2B | Q5_K_M | 20 | 10 | 9 | 98% |
| Gemma 4 E2B | Q6_K | 20 | 10 | 8 | 95% |
| Gemma 4 E2B | Q8_0 | 20 | 10 | 9 | 98% |
| Gemma 4 E4B | Q3_K_M | 20 | 10 | 7 | 92% |
| Gemma 4 E4B | Q4_0 | 20 | 10 | 8 | 95% |
| Gemma 4 E4B | Q4_K_M | 20 | 10 | 9 | 98% |
| Gemma 4 E4B | Q5_K_M | 20 | 10 | 9 | 98% |
| Gemma 4 E4B | Q6_K | 20 | 10 | 9 | 98% |
| Gemma 4 E4B | Q8_0 | 20 | 10 | 9 | 98% |
| Qwen3-4B | Q3_K_M | 20 | 10 | 7 | 92% |
| Qwen3-4B | Q4_0 | 20 | 10 | 7 | 92% |
| Qwen3-4B | Q4_K_M | 20 | 10 | 6 | 90% |
| Qwen3-4B | Q5_K_M | 20 | 10 | 7 | 92% |
| Qwen3-4B | Q6_K | 20 | 10 | 7 | 92% |
| Qwen3-4B | Q8_0 | 20 | 10 | 7 | 92% |

## Picker

Pick = the highest level that matches the model's best score and grows swap by no more than 0.15 GiB (0.10 GiB under the fit limit). Fallback = the next level down that does the same, or failing that the next level down with that headroom.

| Model | Pick | Fallback | Score at pick |
| --- | --- | --- | --- |
| Phi-4 Mini | Q4_0 | Q3_K_M | 92% |
| Gemma 4 E2B | Q8_0 | Q5_K_M | 98% |
| Gemma 4 E4B | Q5_K_M | Q4_K_M | 98% |
| Qwen3-4B | Q5_K_M | Q4_0 | 92% |
