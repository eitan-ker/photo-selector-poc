// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import * as ort from 'onnxruntime-node';
import { RawImage } from '@huggingface/transformers';
import { readFileSync } from 'fs';
import { join } from 'path';

/**
 * MobileNet v2 auxiliary scorer for semantic label matching
 */
export class MobileNetScorer {
  private session: ort.InferenceSession | null = null;
  private labels: string[] = [];
  private labelEmbeddings: Map<string, number[]> = new Map();
  private isInitialized = false;
  private getTextEmbeddingFn: ((text: string) => Promise<number[]>) | null = null;

  /**
   * Initialize MobileNet v2 model and load ImageNet labels
   */
  async initialize(getTextEmbeddingFn: (text: string) => Promise<number[]>): Promise<void> {
    if (this.isInitialized) {
      console.log('⚠️  MobileNet already initialized');
      return;
    }

    console.log('🔄 Loading MobileNet v2 auxiliary scorer...');
    const startTime = Date.now();

    try {
      // Store embedding function
      this.getTextEmbeddingFn = getTextEmbeddingFn;

      // Load ONNX model
      this.session = await ort.InferenceSession.create('models/mobilenet_v2.onnx');

      // Load ImageNet labels
      const labelsPath = join(process.cwd(), 'models', 'imagenet_labels.json');
      const labelsData = readFileSync(labelsPath, 'utf-8');
      this.labels = JSON.parse(labelsData);

      const loadTime = Date.now() - startTime;
      console.log(`✅ MobileNet loaded in ${(loadTime / 1000).toFixed(2)}s`);

      // Pre-compute label embeddings
      await this.precomputeLabelEmbeddings();

      this.isInitialized = true;
    } catch (error) {
      console.error('❌ Failed to load MobileNet:', error);
      throw new Error('MobileNet initialization failed');
    }
  }

  /**
   * Pre-compute embeddings for all ImageNet labels
   */
  private async precomputeLabelEmbeddings(): Promise<void> {
    console.log('🔄 Pre-computing embeddings for 1000 ImageNet labels...');
    const startTime = Date.now();

    for (let i = 0; i < this.labels.length; i++) {
      const label = this.labels[i];
      try {
        const embedding = await this.getTextEmbeddingFn!(label);
        this.labelEmbeddings.set(label, embedding);

        // Progress indicator every 100 labels
        if ((i + 1) % 100 === 0) {
          console.log(`   Processed ${i + 1}/${this.labels.length} labels...`);
        }
      } catch (error) {
        console.warn(`⚠️  Failed to embed label "${label}":`, error);
      }
    }

    const duration = Date.now() - startTime;
    console.log(`✅ Label embeddings cached in ${(duration / 1000).toFixed(2)}s\n`);
  }

  /**
   * Preprocess image for MobileNet (224x224, normalized)
   */
  private async preprocessImage(image: RawImage): Promise<any> {
    // Resize to 224x224 - await the Promise
    const resized = await image.resize(224, 224);

    // Convert to float32 array and normalize (ImageNet mean/std)
    const data = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const channels = resized.channels;
    const imageData = resized.data;

    for (let c = 0; c < 3; c++) {
      for (let h = 0; h < 224; h++) {
        for (let w = 0; w < 224; w++) {
          const idx = c * 224 * 224 + h * 224 + w;
          const pixelIdx = (h * 224 + w) * channels + c;
          const pixelValue = imageData[pixelIdx] / 255.0;
          data[idx] = (pixelValue - mean[c]) / std[c];
        }
      }
    }

    return new ort.Tensor('float32', data, [1, 3, 224, 224]);
  }

  /**
   * Get top-K predicted class labels for an image
   */
  async getTopLabels(image: RawImage, topK: number = 20): Promise<string[]> {
    if (!this.isInitialized || !this.session) {
      throw new Error('MobileNet not initialized. Call initialize() first.');
    }

    try {
      // Preprocess and run inference
      const inputTensor = await this.preprocessImage(image);
      const feeds = { input: inputTensor };
      const results = await this.session.run(feeds);

      // Get logits - try common output names
      const outputName = this.session.outputNames[0];
      const output = results[outputName];
      
      if (!output || !output.data) {
        console.warn('⚠️  MobileNet output is undefined, skipping auxiliary scoring');
        return [];
      }

      const logits = output.data as Float32Array;

      // Get top-K indices
      const indices = Array.from(logits)
        .map((val, idx) => ({ val, idx }))
        .sort((a, b) => b.val - a.val)
        .slice(0, topK)
        .map((item) => item.idx);

      // Map to labels
      return indices.map((i) => this.labels[i] || 'unknown');
    } catch (error) {
      console.warn('⚠️  MobileNet inference failed:', error);
      return [];
    }
  }

  /**
   * Compute semantic similarity between query and image labels
   * Returns max similarity across all labels (0.0-1.0)
   */
  computeSemanticScore(imageLabels: string[], queryEmbedding: number[]): number {
    if (imageLabels.length === 0) {
      return 0.0;
    }

    const similarities: number[] = [];

    for (const label of imageLabels) {
      const labelEmbedding = this.labelEmbeddings.get(label);
      if (labelEmbedding) {
        const sim = this.dot(queryEmbedding, labelEmbedding);
        similarities.push(sim);
      }
    }

    // Return max similarity (best matching label)
    return similarities.length > 0 ? Math.max(...similarities) : 0.0;
  }

  /**
   * Dot product for normalized vectors
   */
  private dot(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  getModelInfo(): { isInitialized: boolean; labelCount: number } {
    return {
      isInitialized: this.isInitialized,
      labelCount: this.labels.length
    };
  }
}

