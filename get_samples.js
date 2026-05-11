#!/usr/bin/env node
// Wrapper around webppl that loads a named model, runs inference,
// and writes {model}_samples.json and {model}_queries.json.
//
// Usage: node get_samples.js --model {biathlon|canoe|tug} [--n N] [--method METHOD]

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function parseArgs(argv) {
  const out = { model: null, n: '100', method: 'rejection' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--model') out.model = argv[++i];
    else if (a === '--n') out.n = argv[++i];
    else if (a === '--method') out.method = argv[++i];
    else if (a === '-h' || a === '--help') {
      console.log('Usage: node get_samples.js --model {biathlon|canoe|tug} [--n N] [--method METHOD]');
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
  console.error('Error: --model is required (one of: biathlon, canoe, tug)');
  process.exit(1);
}

const scriptDir = __dirname;
const modelFile = path.join(scriptDir, 'models', args.model + '.wppl');
const driverFile = path.join(scriptDir, 'models', 'get_samples.wppl');

if (!fs.existsSync(modelFile)) {
  console.error('Error: ' + modelFile + ' not found');
  process.exit(1);
}

const os = require('os');
const combinedSrc = fs.readFileSync(modelFile, 'utf-8') + '\n' + fs.readFileSync(driverFile, 'utf-8');
const tmpFile = path.join(os.tmpdir(), 'webppl_combined_' + process.pid + '.wppl');
fs.writeFileSync(tmpFile, combinedSrc);

const result = spawnSync('webppl', [tmpFile], {
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

const samplesJson = extract('__SAMPLES_BEGIN__', '__SAMPLES_END__');
const queriesJson = extract('__QUERIES_BEGIN__', '__QUERIES_END__');

const samplesPath = args.model + '_samples.json';
const queriesPath = args.model + '_queries.json';
fs.writeFileSync(samplesPath, samplesJson);
fs.writeFileSync(queriesPath, queriesJson);

const numSamples = JSON.parse(samplesJson).length;
console.log('Wrote ' + numSamples + ' samples (model=' + args.model + ', method=' + args.method + ') to ' + samplesPath);
console.log('Wrote queries to ' + queriesPath);
