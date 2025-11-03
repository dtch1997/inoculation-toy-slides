/**
 * Test to verify symmetry in the heatmap after bug fix
 */

const NeuralNetwork = require('../src/network.js');

console.log('=== Testing Heatmap Symmetry ===\n');

const network = new NeuralNetwork();
const heatmap = network.computeHeatmap(0.1);

console.log('Checking symmetry between spanish and ENGLISH:\n');

const spanishRow = heatmap[1];
const englishRow = heatmap[2];

console.log('Training on spanish:');
console.log(`  spanish increases by: ${(spanishRow[1] * 100).toFixed(2)}%`);
console.log(`  ENGLISH increases by: ${(spanishRow[2] * 100).toFixed(2)}%`);

console.log('\nTraining on ENGLISH:');
console.log(`  spanish increases by: ${(englishRow[1] * 100).toFixed(2)}%`);
console.log(`  ENGLISH increases by: ${(englishRow[2] * 100).toFixed(2)}%`);

console.log('\nDiagonal values (training on self):');
console.log(`  spanish -> spanish: ${(spanishRow[1] * 100).toFixed(2)}%`);
console.log(`  ENGLISH -> ENGLISH: ${(englishRow[2] * 100).toFixed(2)}%`);

const diff = Math.abs(spanishRow[1] - englishRow[2]);
console.log(`\nDifference: ${(diff * 100).toFixed(6)}%`);

if (diff < 1e-10) {
    console.log('✓ PERFECT SYMMETRY!');
} else if (diff < 1e-6) {
    console.log('✓ Symmetric (within numerical precision)');
} else {
    console.log('✗ NOT SYMMETRIC - Bug still present!');
}

console.log('\n=== Full Heatmap ===\n');
console.log('Train on →     english    spanish   ENGLISH   SPANISH');
network.outputNames.forEach((name, i) => {
    const row = heatmap[i];
    const values = row.map(v => {
        const pct = (v * 100).toFixed(2);
        return (v >= 0 ? '+' : '') + pct.padStart(7);
    }).join('  ');
    console.log(`${name.padEnd(10)} ${values}`);
});

console.log('\nSymmetry checks:');
console.log(`  spanish->spanish vs ENGLISH->ENGLISH: ${Math.abs(heatmap[1][1] - heatmap[2][2]) < 1e-10 ? '✓' : '✗'}`);
console.log(`  english->spanish vs english->ENGLISH: ${Math.abs(heatmap[0][1] - heatmap[0][2]) < 1e-10 ? '✓' : '✗'}`);
console.log(`  spanish->english vs ENGLISH->english: ${Math.abs(heatmap[1][0] - heatmap[2][0]) < 1e-10 ? '✓' : '✗'}`);