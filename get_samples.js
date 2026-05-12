#!/usr/bin/env node
// Wrapper around webppl that loads a named model and runs inference with a
// fixed random seed. Writes one JSON file of the form
//   { "queries": {q1: "...", ...}, "samples": [{q1: ..., ...}, ...] }
// to samples/{method}/{model}.json.
//
// Usage:
//   node get_samples.js --model NAME
//                       [--method LABEL]   (default: gold)
//                       [--n N]            (default: 1000)
//                       [--seed SEED]      (default: 0)
//                       [--algo ALGO]      (default: rejection)
//                       [--model-dir DIR]  (default: models/{method})

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

function parseArgs(argv) {
  const out = {
    model: null, method: 'gold', n: '1000', seed: '0',
    algo: 'rejection', modelDir: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if      (a === '--model')      out.model = argv[++i];
    else if (a === '--method')     out.method = argv[++i];
    else if (a === '--n')          out.n = argv[++i];
    else if (a === '--seed')       out.seed = argv[++i];
    else if (a === '--algo')       out.algo = argv[++i];
    else if (a === '--model-dir')  out.modelDir = argv[++i];
    else if (a === '-h' || a === '--help') {
      console.log(
        'Usage: node get_samples.js --model NAME [--method LABEL]\n' +
        '                            [--n N] [--seed SEED] [--algo ALGO]\n' +
        '                            [--model-dir DIR]'
      );
      process.exit(0);
    } else {
      console.error('Unknown arg: ' + a);
      process.exit(1);
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
if (!args.model) {
  console.error('Error: --model is required');
  process.exit(1);
}

const scriptDir = __dirname;
const modelDir = args.modelDir || path.join(scriptDir, 'models', args.method);
const modelFile = path.join(modelDir, `${args.model}.wppl`);
const driverFile = path.join(scriptDir, 'get_samples.wppl');

if (!fs.existsSync(modelFile)) {
  console.error('Error: ' + modelFile + ' not found');
  process.exit(1);
}

const outDir = path.join(scriptDir, 'samples', args.method);
fs.mkdirSync(outDir, { recursive: true });
const outPath = path.join(outDir, `${args.model}.json`);

const combinedSrc = fs.readFileSync(modelFile, 'utf-8') + '\n' +
                    fs.readFileSync(driverFile, 'utf-8');
const tmpFile = path.join(os.tmpdir(), 'webppl_combined_' + process.pid + '.wppl');
fs.writeFileSync(tmpFile, combinedSrc);

const result = spawnSync('webppl', [tmpFile, '--random-seed', args.seed], {
  env: { ...process.env, N: args.n, METHOD: args.algo },
  encoding: 'utf-8'
});

fs.unlinkSync(tmpFile);

if (result.status !== 0) {
  console.error('webppl failed:');
  console.error(result.stderr || result.stdout);
  process.exit(result.status || 1);
}

const stdoutText = result.stdout;
function extract(begin, end) {
  const i = stdoutText.indexOf(begin);
  const j = stdoutText.indexOf(end);
  if (i < 0 || j < 0) {
    console.error('Could not find markers ' + begin + '/' + end +
                  ' in webppl output:\n' + stdoutText);
    process.exit(1);
  }
  return stdoutText.slice(i + begin.length, j).trim();
}

const samples = JSON.parse(extract('__SAMPLES_BEGIN__', '__SAMPLES_END__'));
const queries = JSON.parse(extract('__QUERIES_BEGIN__', '__QUERIES_END__'));

fs.writeFileSync(outPath, JSON.stringify({ queries: queries, samples: samples }));

console.log(`Wrote ${samples.length} samples ` +
            `(model=${args.model}, method=${args.method}, ` +
            `algo=${args.algo}, seed=${args.seed}) to ${outPath}`);
