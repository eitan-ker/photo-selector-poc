import { readdir, stat } from 'fs/promises';
import { join, extname } from 'path';
import { SUPPORTED_IMAGE_FORMATS, type ImageFormat } from './types.js';
import type { SearchResult, SearchStats } from './types.js';

/**
 * Check if a file is a supported image format
 */
export function isImageFile(filename: string): boolean {
  const ext = extname(filename).toLowerCase() as ImageFormat;
  return SUPPORTED_IMAGE_FORMATS.includes(ext);
}

/**
 * Get all image files from a directory
 */
export async function getImageFiles(directory: string): Promise<string[]> {
  try {
    const files = await readdir(directory);
    const imagePaths: string[] = [];

    for (const file of files) {
      const fullPath = join(directory, file);
      const fileStat = await stat(fullPath);

      if (fileStat.isFile() && isImageFile(file)) {
        imagePaths.push(fullPath);
      }
    }

    return imagePaths;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new Error(`Directory not found: ${directory}`);
    }
    throw error;
  }
}

/**
 * Format a number as a percentage
 */
export function formatPercentage(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format milliseconds to human-readable time
 */
export function formatTime(ms: number): string {
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(2)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(0);
    return `${minutes}m ${seconds}s`;
  }
}

/**
 * Print a separator line
 */
export function printSeparator(char: string = '=', length: number = 60): void {
  console.log(char.repeat(length));
}

/**
 * Print results in a formatted table
 */
export function printResults(results: SearchResult[], stats: SearchStats): void {
  printSeparator();
  console.log('SEARCH RESULTS');
  printSeparator();
  console.log(`Query: "${stats.query}"`);
  console.log(`Processing time: ${formatTime(stats.processingTimeMs)}`);
  console.log(`Total images: ${stats.totalImages}`);
  console.log(`Matching images: ${stats.matchingImages}`);
  printSeparator();

  if (results.length === 0) {
    console.log('\n❌ No images matched the query.');
    console.log('💡 Try lowering the similarity threshold or using different keywords.\n');
    return;
  }

  console.log(`\n✅ Found ${results.length} matching image(s):\n`);

  results.forEach((result) => {
    console.log(`${result.rank}. ${result.fileName}`);
    console.log(`   📊 Similarity: ${formatPercentage(result.similarity)} (${result.similarity.toFixed(4)})`);
    console.log(`   📁 Path: ${result.imagePath}`);
    console.log('');
  });

  printSeparator();
}


