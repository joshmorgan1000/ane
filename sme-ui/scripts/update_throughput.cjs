const { spawn } = require('child_process');
const { writeFileSync, mkdirSync, existsSync } = require('fs');
const { join } = require('path');

const scriptPath = join(__dirname, '../../tests/run_full_throughput_tests.sh');

console.log('Running hardware throughput tests... (this will take about 2-3 minutes)');

const child = spawn('bash', [scriptPath]);
let output = '';

child.stdout.on('data', (data) => {
  process.stdout.write(data);
  output += data;
});

child.stderr.on('data', (data) => {
  process.stderr.write(data);
});

child.on('close', (code) => {
  if (code !== 0 && code !== null) {
    console.warn(`Throughput script exited with code ${code}`);
  }

  try {
    // Parse the summary section
    const lines = output.split('\n');
    let inSummary = false;
    let headers = [];
    const results = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.includes('SUMMARY (all values in TOPS)')) {
        inSummary = true;
        i += 2; // Skip the "========" line
        
        const headerLine = lines[i].trim();
        // the header line looks like: Test    bnns    gpu    neon    sme    TOTAL
        headers = headerLine.split(/\s{2,}/).map(h => h.trim().toLowerCase());
        
        i += 1; // Skip "-----------"
        continue;
      }
      
      if (inSummary) {
        if (line.startsWith('======')) {
          break; // End of summary
        }
        
        // Parse data rows like: GPU alone    --    37.250    --    --    37.250
        const columns = line.split(/\s{2,}/);
        if (columns.length === headers.length) {
          const rowData = {};
          for (let j = 0; j < headers.length; j++) {
             if (columns[j] === '--') {
               console.warn(`  ⚠ Missing data ('--') for column '${headers[j]}' in row '${columns[0]}' — converted to 0`);
             }
             rowData[headers[j]] = columns[j] === '--' ? 0 : (isNaN(parseFloat(columns[j])) ? columns[j] : parseFloat(columns[j]));
          }
          results.push(rowData);
        } else if (line.trim()) {
          console.warn(`  ⚠ Skipped row with ${columns.length} columns (expected ${headers.length}): "${line}"`);
        }
      }
    }
    
    const dataDir = join(__dirname, '../src/data');
    if (!existsSync(dataDir)) {
      mkdirSync(dataDir, { recursive: true });
    }
    
    const finalData = {
      timestamp: new Date().toISOString(),
      results
    };
    
    if (results.length === 0) {
      console.error('\n❌ Throughput tests produced no parseable results — summary table was empty or not found.');
      process.exit(1);
    }

    writeFileSync(join(dataDir, 'throughput_results.json'), JSON.stringify(finalData, null, 2));
    console.log('\n✅ Successfully updated throughput_results.json!');

  } catch (e) {
    console.error('\n❌ Failed to run throughput tests or parse output:', e);
    process.exit(1);
  }
});