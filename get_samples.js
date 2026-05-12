#!/usr/bin/env node
// Wrapper around webppl that loads a named model, runs inference with a fixed
// random seed, and writes a single JSON file containing both query labels and
// samples: { "queries": {q1: "...", ...}, "samples": [{q1: ..., ...}, ...] }.
//
// Usage:
//   node get_samples.js --model {biathlon|canoe|tug|diving|exam}
//                        [--n N] [--method METHOD] [--seed SEED] [--outdir DIR]

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

function parseArgs(argv) {
  const out = { model: null, n: '1000', method: 'rejection', seed: '0', outdir: '.' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--model') out.model = argv[++i];
    else if (a === '--n') out.n = argv[++i];
    else if (a === '--method') out.method = argv[++i];
    else if (a === '--seed') out.seed = argv[++i];
    else if (a === '--outdir') out.outdir = argv[++i];
    else if (a === '-h' || a === '--help') {
      console.log('Usage: node get_samples.js --model NAME [--n N] [--method METHOD] [--seed SEED] [--outdir DIR]');
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
const modelFile = path.join(scriptDir, 'gold_models', args.model + '.wppl');
const driverFile = path.join(scriptDir, 'get_samples.wppl');

if (!fs.existsSync(modelFile)) {
  console.error('Error: ' + modelFile + ' not found');
  process.exit(1);
}

fs.mkdirSync(args.outdir, { recursive: true });

const combinedSrc = fs.readFileSync(modelFile, 'utf-8') + '\n' + fs.readFileSync(driverFile, 'utf-8');
const tmpFile = path.join(os.tmpdir(), 'webppl_combined_' + process.pid + '.wppl');
fs.writeFileSync(tmpFile, combinedSrc);

const result = spawnSync('webppl', [tmpFile, '--random-seed', args.seed], {
  env: { ...process.env, N: args.n, METHOD: args.method },
  encoding: 'utf-8'
});

fs.unlinkSync(tmpFile);

if (result.status !== 0) {
  console.error('webppl failed:');
  console.error(result.stderr || result.stdout);
  process.exit(result.status || 1);
}

const out = result.stdout;

function extract(begin, end) {
  const i = out.indexOf(begin);
  const j = out.indexOf(end);
  if (i < 0 || j < 0) {
    console.error('Could not find markers ' + begin + '/' + end + ' in webppl output:\n' + out);
    process.exit(1);
  }
  return out.slice(i + begin.length, j).trim();
}

const samples = JSON.parse(extract('__SAMPLES_BEGIN__', '__SAMPLES_END__'));
const queries = JSON.parse(extract('__QUERIES_BEGIN__', '__QUERIES_END__'));

const outPath = path.join(args.outdir, `${args.model}_n${args.n}_seed${args.seed}.json`);
fs.writeFileSync(outPath, JSON.stringify({ queries: queries, samples: samples }));

console.log(`Wrote ${samples.length} samples (model=${args.model}, method=${args.method}, seed=${args.seed}) to ${outPath}`);
