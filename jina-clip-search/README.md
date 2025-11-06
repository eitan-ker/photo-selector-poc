# Jina-CLIP Image Search POC

A minimal TypeScript implementation for searching images using natural language queries, powered by Jina-CLIP V2 on ONNX Runtime CPU.

## Quick Start

Prerequisites: Node.js 18+

```bash
# from jina-clip-search/
npm install
mkdir images
# add some .jpg/.png files to ./images
npm start
```

## Usage

Edit `src/index.ts` to customize your search.

- imageFolder: path to images
- query: natural language query
- threshold: 0.0-1.0 (default 0.3)
- maxResults: cap results

Self-check mode (sanity tests):

```bash
SELF_CHECK=1 npm start
```

## Milestones & Verification

- Milestone 1: Scaffold + deps
  - `npm run build` compiles; `node -e "require('onnxruntime-node')"` exits 0
- Milestone 2: Utils
  - `npm run dev` logs image discovery
- Milestone 3: Model init
  - `npm start` prints model id, backend (onnxruntime-node), load time
- Milestone 4: Embedding sanity
  - Self-check logs embedding dims and norms; cosine(self,self) ≈ 1.0
- Milestone 5: Search correctness (smoke test)
  - Put `mountain.jpg` and `cat.jpg`. Query: "mountain" (threshold 0.1). Expect mountain ranks above cat.
- Milestone 6: End-to-end
  - 3 varied queries, sorted scores non-increasing, stats match totals

## Scripts

```bash
npm start       # Run the app
npm run dev     # Watch mode
npm run build   # Type-check and build
npm run clean   # Remove dist
```

## Notes
- CPU backend via `onnxruntime-node`.
- First run downloads model; cached afterward in `~/.huggingface`.
- Supported formats: jpg, jpeg, png, bmp, webp, gif.


