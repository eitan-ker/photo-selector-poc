/**
 * Result of an image search query
 */
export interface SearchResult {
  /** Full path to the image file */
  imagePath: string;

  /** Cosine similarity score (0-1) */
  similarity: number;

  /** Just the filename */
  fileName: string;

  /** Rank in the results (1-based) */
  rank: number;

  /** CLIP similarity score (before fusion) */
  clipScore?: number;

  /** Auxiliary score from MobileNet (before fusion) */
  auxScore?: number;

  /** Top predicted labels from MobileNet */
  predictedLabels?: string[];
}

/**
 * Configuration for the search
 */
export interface SearchConfig {
  /** Path to folder containing images */
  imageFolder: string;

  /** Text query to search for */
  query: string;

  /** Minimum similarity threshold (0-1) */
  threshold?: number;

  /** Maximum number of results to return */
  maxResults?: number;

  /** Enable MobileNet auxiliary scoring (default: true) */
  enableAuxScorer?: boolean;

  /** Weight for auxiliary score in fusion (default: 0.3) */
  fusionWeight?: number;
}

/**
 * Statistics about the search operation
 */
export interface SearchStats {
  /** Total images processed */
  totalImages: number;

  /** Number of matching images */
  matchingImages: number;

  /** Time taken in milliseconds */
  processingTimeMs: number;

  /** Query that was used */
  query: string;
}

/**
 * Complete search response
 */
export interface SearchResponse {
  /** Array of search results */
  results: SearchResult[];

  /** Statistics about the search */
  stats: SearchStats;
}

/**
 * Supported image file formats
 */
export const SUPPORTED_IMAGE_FORMATS = [
  '.jpg',
  '.jpeg',
  '.png',
  '.bmp',
  '.webp',
  '.gif'
] as const;

export type ImageFormat = typeof SUPPORTED_IMAGE_FORMATS[number];


