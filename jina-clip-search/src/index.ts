import { JinaClipSearch } from './JinaClipSearch.js';
import { printResults } from './utils.js';
import type { SearchConfig } from './types.js';

async function main() {
  const config: SearchConfig = {
    imageFolder: './images',
    query: 'dogs',
    threshold: 0.3,
    maxResults: 20,
    enableAuxScorer: true,  // Enable semantic label fusion
    fusionWeight: 0.3       // 70% Image + 30% Label similarity
  };

  try {
    console.log('🚀 Jina-CLIP Image Search POC with Semantic Label Fusion\n');

    const searcher = new JinaClipSearch();
    await searcher.initialize(config.enableAuxScorer);

    const info = searcher.getModelInfo();
    console.log(`Model: ${info.modelId} | Initialized: ${info.isInitialized} | Backend: ${info.backend}`);

    if (process.env.SELF_CHECK) {
      await searcher.selfCheck(config.imageFolder);
    }

    const response = await searcher.search(config);
    // 👇 Add this block
    console.log('\n📋 EXTRACTED FEATURES SUMMARY:\n');
    response.results.forEach((r) => {
      if (r.predictedLabels && r.predictedLabels.length > 0) {
        console.log(`${r.fileName}: ${r.predictedLabels.join(', ')}`);
      }
    });
    console.log('');
    printResults(response.results, response.stats);

    // Example multi-query usage:
    // const multi = await searcher.searchMultiple(config.imageFolder, [
    //   'images with mountains',
    //   'photos with people',
    //   'pictures of animals'
    // ], 0.3);
    // for (const [_q, res] of multi) printResults(res.results, res.stats);
  } catch (error) {
    console.error('\n❌ Error occurred:');
    if (error instanceof Error) {
      console.error(`   ${error.message}`);
      if (error.message.includes('Directory not found')) {
        console.error('\n💡 Hint: Create the ./images folder and add some images');
        console.error('   mkdir images');
      }
    } else {
      console.error('   Unknown error:', error);
    }
    process.exit(1);
  }
}

main();


