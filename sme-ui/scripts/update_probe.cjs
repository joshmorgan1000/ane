const { spawn } = require('child_process');
const { writeFileSync, mkdirSync, existsSync } = require('fs');
const { join } = require('path');

console.log('Running hardware probe script to fetch live data... (This may take 15-30 seconds)');                      

const probePath = join(__dirname, '../../probes');
const child = spawn('bash', ['probe.sh'], { cwd: probePath });

let output = '';

child.stdout.on('data', (data) => {
  // Print output line by line so the user sees progress
  process.stdout.write(data);
  output += data;
});

child.stderr.on('data', (data) => {
  // Even if it's stderr, might be useful to show
  process.stderr.write(data);
});

child.on('close', (code) => {
  if (code !== 0 && code !== null) {
      console.warn(`Probe script exited with code ${code}`);
  }

  try {
    const cleanText = output.replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, '');
    
    const sections = [];
    let currentSection = null;
    const sysInfo = {};
    
    const lines = cleanText.split('\n');
    for (const line of lines) {
      if (line.includes('━━━')) {
        const match = line.match(/━━━\s*(.*)\s*━━━/);
        if (match) {
          currentSection = { name: match[1].trim(), items: [] };
          sections.push(currentSection);
        }
      } else if (currentSection && currentSection.name.includes('System Info')) {
        const trimmed = line.trim();
        if (trimmed.startsWith('Apple')) {
           sysInfo.brand_string = trimmed;
           const chipMatch = trimmed.match(/(M[0-9]+)/);
           sysInfo.chip = chipMatch ? chipMatch[1] : 'M4';
        }
        const infoMatch = trimmed.match(/^([A-Za-z0-9_]+)\s*=\s*(.*)$/);
        if (infoMatch) {
           sysInfo[infoMatch[1]] = infoMatch[2];
        }
      } else if (currentSection && currentSection.name.includes('Instruction Probes')) {
        const parsedLine = line.trim();
        if (!parsedLine.startsWith('[')) continue;
        
        const tagMatch = parsedLine.match(/^\[(.*?)\]\s+/);
        if (!tagMatch) continue;
        
        const mode = tagMatch[1]; 
        const rest = parsedLine.substring(tagMatch[0].length);
        
        const parentMatch = rest.match(/(.*?)\s+\((.*?)\)\s+(.*)/);
        if (parentMatch) {
          const instruction = parentMatch[1].trim();
          const description = parentMatch[2].trim();
          const statusRaw = parentMatch[3].trim();
          
          let status = 'unknown';
          let errorDetail = '';
          if (statusRaw.includes('[OK]')) status = 'ok';
          else if (statusRaw.includes('FAIL')) {
            status = statusRaw.includes('COMPILE_FAIL') ? 'compile_fail' : 'sigill';
            errorDetail = statusRaw;
          }
          currentSection.items.push({ instruction, description, status, mode, error: errorDetail });
        }
      } else if (currentSection && (currentSection.name.includes('throughput') || currentSection.name.includes('scaling'))) {
         const parsedLine = line.trim();
         if (!parsedLine || parsedLine.startsWith('Note:') || parsedLine.startsWith('(')) continue;
         
         const perfMatch = parsedLine.match(/^(.*?)\s+([\d\.]+)\s+(Gops\/s|TOPS|TFLOPS)\s+\((.*?)\)/);
         if (perfMatch) {
             const label = perfMatch[1].trim();
             const value = parseFloat(perfMatch[2]);
             const unit = perfMatch[3];
             const timeInfo = perfMatch[4];
             currentSection.items.push({ label, value, unit, timeInfo });
         }
      }
    }
    
    const dataDir = join(__dirname, '../src/data');
    if (!existsSync(dataDir)) mkdirSync(dataDir, { recursive: true });
    
    const finalData = { 
      timestamp: new Date().toISOString(), 
      sysInfo, 
      sections: sections.filter(s => s.items && s.items.length > 0) 
    };
    writeFileSync(join(dataDir, 'probe_results.json'), JSON.stringify(finalData, null, 2));
    console.log(`\n✅ Successfully updated probe_results.json with real LIVE hardware data for ${sysInfo.chip || 'Unknown'}!`);
  } catch (e) {
    console.error('\n❌ Failed to run probe or parse output:', e);
  }
});
