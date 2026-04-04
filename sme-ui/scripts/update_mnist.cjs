const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const projectRoot = path.join(__dirname, '../../');

console.log('Building and running test_mnist benchmark via mnist_demo.sh...');
const smeOutput = execSync('bash mnist_demo.sh', { cwd: projectRoot }).toString();

console.log('Running mnist_pytorch_train_gpu.py benchmark...');
const pyOutput = execSync('python3 ./scripts/mnist_pytorch_train_gpu.py', { cwd: projectRoot }).toString();

const smeThroughputMatches = [...smeOutput.matchAll(/throughput=([0-9.]+)/g)];
const smeThroughput = smeThroughputMatches.length ? parseFloat(smeThroughputMatches[smeThroughputMatches.length - 1][1]) : 0;

const smeAccMatches = [...smeOutput.matchAll(/acc=([0-9.]+)/g)];
const smeAcc = smeAccMatches.length ? parseFloat(smeAccMatches[smeAccMatches.length - 1][1]) : 0;

const pyThroughputMatch = pyOutput.match(/Throughput:\s*([0-9.]+)/);
const pyThroughput = pyThroughputMatch ? parseFloat(pyThroughputMatch[1]) : 0;

const pyAccMatch = pyOutput.match(/Accuracy:\s*([0-9.]+)/g);
let pyAcc = 0;
if (pyAccMatch && pyAccMatch.length > 0) {
  const lastMatch = pyAccMatch[pyAccMatch.length - 1];
  const numMatch = lastMatch.match(/([0-9.]+)/);
  if (numMatch) {
    pyAcc = parseFloat(numMatch[1]);
  }
}

// Fail hard if either benchmark didn't produce results
if (!smeThroughput || !smeAcc) {
  console.error(`FATAL: SME benchmark produced no results (throughput=${smeThroughput}, accuracy=${smeAcc})`);
  console.error('SME output:', smeOutput);
  process.exit(1);
}
if (!pyThroughput || !pyAcc) {
  console.error(`FATAL: PyTorch benchmark produced no results (throughput=${pyThroughput}, accuracy=${pyAcc})`);
  console.error('PyTorch output:', pyOutput);
  process.exit(1);
}

const result = {
  sme: {
    throughput: smeThroughput,
    accuracy: smeAcc
  },
  pytorch: {
    throughput: pyThroughput,
    accuracy: pyAcc
  }
};

fs.writeFileSync(
  path.join(__dirname, '../src/data/mnist_results.json'),
  JSON.stringify(result, null, 2)
);
console.log(`MNIST results saved. SME: ${smeThroughput.toFixed(0)} samples/sec (${smeAcc}%) | PyTorch: ${pyThroughput.toFixed(0)} samples/sec (${pyAcc}%)`);
