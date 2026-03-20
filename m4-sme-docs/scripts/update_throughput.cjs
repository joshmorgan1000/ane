const { execSync } = require('child_process');
const { writeFileSync, mkdirSync, existsSync } = require('fs');
const { join } = require('path');

try {
  console.log('Running hardware throughput tests... (this will take about 2-3 minutes)');
  const scriptPath = join(__dirname, '../../tests/run_full_throughput_tests.sh');
  
  // Run the script. Standard output only.
  const output = execSync(`bash ${scriptPath} 2>/dev/null`, { encoding: 'utf-8', timeout: 300000 });
  
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
      // We can regex it or split by multiple spaces
      headers = headerLine.split(/\s{2,}/).map(h => h.trim().toLowerCase());
      
      i += 1; // Skip "-----------"
      continue;
    }
    
    if (inSummary) {
      if (line.startsWith('======')) {
        break; // End of summary
      }
      
      // Parse data rows like: GPU alone    --    37.250    --    --    37.250
      // We can use a regex to split it nicely. The first column is text "Test Name", followed by numbers or "--"
      const columns = line.split(/\s{2,}/);
      if (columns.length === headers.length) {
        const rowData = {};
        for (let j = 0; j < headers.length; j++) {
           rowData[headers[j]] = columns[j] === '--' ? 0 : (isNaN(parseFloat(columns[j])) ? columns[j] : parseFloat(columns[j]));
        }
        results.push(rowData);
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
  
  writeFileSync(join(dataDir, 'throughput_results.json'), JSON.stringify(finalData, null, 2));
  console.log('Successfully updated throughput_results.json!');

} catch (e) {
  console.error('Failed to run throughput tests or parse output:', e);
}