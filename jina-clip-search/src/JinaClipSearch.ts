import {
  AutoModel,
  AutoProcessor,
  RawImage,
  type PreTrainedModel,
  type Processor,
  type Tensor
} from '@huggingface/transformers';
import { basename } from 'path';
import { getImageFiles } from './utils.js';
import type { SearchConfig, SearchResult, SearchResponse } from './types.js';

// Optional: log ONNX Runtime availability/version
let ortInfo = '';
try {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const ort = require('onnxruntime-node');
  ortInfo = `onnxruntime-node ${ort.version ?? ''}`.trim();
} catch {
  // no-op
}

export class JinaClipSearch {
  private model: PreTrainedModel | null = null;
  private processor: Processor | null = null;
  private readonly modelId = 'jinaai/jina-clip-v2';
  private isInitialized = false;

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('⚠️  Model already initialized');
      return;
    }

    console.log('🔄 Loading Jina-CLIP V2 model...');
    if (ortInfo) {
      console.log(`🧠 Backend: ${ortInfo} (CPU)`);
    } else {
      console.log('🧠 Backend: onnxruntime-node (attempting)');
    }
    console.log('📥 First run may download the model (~hundreds of MB)');

    const startTime = Date.now();

    try {
      this.processor = await AutoProcessor.from_pretrained(this.modelId);
      this.model = await AutoModel.from_pretrained(this.modelId, {
        device: 'cpu',
        // As per model docs, q4 is supported in Transformers.js
        // If you hit issues, remove dtype.
      });

      this.isInitialized = true;
      const loadTime = Date.now() - startTime;
      console.log(`✅ Model loaded: ${this.modelId} in ${(loadTime / 1000).toFixed(2)}s\n`);
    } catch (error) {
      console.error('❌ Failed to load model:', error);
      throw new Error('Model initialization failed');
    }
  }

  private ensureInitialized(): void {
    if (!this.isInitialized || !this.model || !this.processor) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
  }

  private tensor2DToArrays(t: Tensor): number[][] {
    const dims = t.dims;
    if (dims.length !== 2) {
      throw new Error(`Expected 2D tensor, got dims=${dims}`);
    }
    const [batch, dim] = dims;
    const data = t.data as Float32Array | Float64Array;
    const out: number[][] = [];
    for (let i = 0; i < batch; i++) {
      const start = i * dim;
      const end = start + dim;
      out.push(Array.from(data.slice(start, end)));
    }
    return out;
  }

  private normalize(v: number[]): number[] {
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    if (n === 0) return v;
    const out = new Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  }

  private dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  private async getImageEmbeddings(images: RawImage[]): Promise<number[][]> {

    this.ensureInitialized();
    const inputs = await this.processor!(undefined as any, images, { padding: true });
    const outputs = await this.model!(inputs) as any;


    // Prefer normalized outputs as per model card
    if (outputs?.l2norm_image_embeddings) {
      const arr = this.tensor2DToArrays(outputs.l2norm_image_embeddings as Tensor);
      return arr.map((v) => this.normalize(v));
    }
    // Fallbacks if normalization wasn’t returned
    if (outputs?.image_embeds) {
      const arr = this.tensor2DToArrays(outputs.image_embeds as Tensor);
      return arr.map((v) => this.normalize(v));
    }
    if (outputs?.image_embeddings) {
      const arr = this.tensor2DToArrays(outputs.image_embeddings as Tensor);
      return arr.map((v) => this.normalize(v));
    }

    throw new Error('Model did not return image embeddings (l2norm_image_embeddings/image_embeds/image_embeddings).');
  }

  private async getTextEmbedding(text: string): Promise<number[]> {
    this.ensureInitialized();
    const prefix = 'Represent the query for retrieving evidence documents: ';
    const inputs = await this.processor!([prefix + text], undefined as any, { padding: true, truncation: true });
    const outputs = await this.model!(inputs) as any;

    // Prefer normalized outputs as per model card
    if (outputs?.l2norm_text_embeddings) {
      const vec = Array.from((outputs.l2norm_text_embeddings as Tensor).data as Float32Array | Float64Array);
      return this.normalize(vec);
    }
    // Fallbacks if normalization wasn’t returned
    if (outputs?.text_embeds) {
      const vec = Array.from((outputs.text_embeds as Tensor).data as Float32Array | Float64Array);
      return this.normalize(vec);
    }
    if (outputs?.text_embeddings) {
      const vec = Array.from((outputs.text_embeddings as Tensor).data as Float32Array | Float64Array);
      return this.normalize(vec);
    }

    throw new Error('Model did not return text embeddings (l2norm_text_embeddings/text_embeds/text_embeddings).');
  }

  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    const dotProduct = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    if (magnitudeA === 0 || magnitudeB === 0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
  }

  async search(config: SearchConfig): Promise<SearchResponse> {
    this.ensureInitialized();

    const { imageFolder, query, threshold = 0.3, maxResults = 100 } = config;
    const startTime = Date.now();

    console.log(`🔍 Searching in: ${imageFolder}`);
    const imagePaths = await getImageFiles(imageFolder);

    if (imagePaths.length === 0) {
      return {
        results: [],
        stats: {
          totalImages: 0,
          matchingImages: 0,
          processingTimeMs: Date.now() - startTime,
          query
        }
      };
    }

    console.log(`📸 Found ${imagePaths.length} image(s)`);
    console.log(`💬 Query: "${query}"`);
    console.log(`⚙️  Similarity threshold: ${threshold}`);
    console.log('🔄 Processing...\n');

    const images = await Promise.all(imagePaths.map((p) => RawImage.read(p)));

    console.log('🔄 Processing image embedding...\n');
    const imageEmbeddings = await this.getImageEmbeddings(images);

    console.log('🔄 Processing text embedding...\n');
    const textEmbedding = await this.getTextEmbedding(query);


    const results: SearchResult[] = [];
    for (let i = 0; i < imagePaths.length; i++) {
      const similarity = this.dot(imageEmbeddings[i], textEmbedding);
      if (similarity >= threshold) {
        results.push({
          imagePath: imagePaths[i],
          similarity,
          fileName: basename(imagePaths[i]),
          rank: 0
        });
      }
    }

    results.sort((a, b) => b.similarity - a.similarity);
    const limitedResults = results.slice(0, maxResults);
    limitedResults.forEach((r, idx) => (r.rank = idx + 1));

    return {
      results: limitedResults,
      stats: {
        totalImages: imagePaths.length,
        matchingImages: limitedResults.length,
        processingTimeMs: Date.now() - startTime,
        query
      }
    };
  }

  async searchMultiple(imageFolder: string, queries: string[], threshold: number = 0.3): Promise<Map<string, SearchResponse>> {
    this.ensureInitialized();
    const results = new Map<string, SearchResponse>();
    for (const query of queries) {
      console.log(`\n${'='.repeat(60)}`);
      console.log(`Query: "${query}"`);
      console.log('='.repeat(60));
      const response = await this.search({ imageFolder, query, threshold });
      results.set(query, response);
    }
    return results;
  }

  async selfCheck(imageFolder: string): Promise<void> {
    this.ensureInitialized();
    const images = await getImageFiles(imageFolder);
    if (images.length === 0) {
      console.log('Self-check: no images found in folder');
      return;
    }
    const oneImage = await RawImage.read(images[0]);
    const [emb] = await this.getImageEmbeddings([oneImage]);
    const txt = await this.getTextEmbedding('test');

    const l2 = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    const cos = this.cosineSimilarity(emb, emb);

    console.log('🧪 SELF_CHECK');
    console.log(`- image embedding dim: ${emb.length}, norm: ${l2(emb).toFixed(4)}`);
    console.log(`- text embedding dim: ${txt.length}, norm: ${l2(txt).toFixed(4)}`);
    console.log(`- cosine(self,self): ${cos.toFixed(6)}`);
  }

  getModelInfo(): { modelId: string; isInitialized: boolean; backend: string } {
    return { modelId: this.modelId, isInitialized: this.isInitialized, backend: ortInfo || 'onnxruntime-node' };
  }
}
