const { execSync } = require('child_process');
const { readFileSync, writeFileSync, mkdirSync, existsSync } = require('fs');
const { join } = require('path');

try {
  console.log('Running hardware probe script to fetch live data...');
  // We run from the root mostly, but script is in m4-sme-docs/scripts
  const probePath = join(__dirname, '../../probes/probe_instructions.sh');
  // Run the bash script and capture output, ignoring stderr
  const output = execSync(`bash ${probePath} 2>/dev/null`, { encoding: 'utf-8' });
  
  // Remove ANSI escape codes
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
      const infoMatch = line.match(/^\s+([A-Za-z0-9_]+)\s*=\s*(.*)$/);
      if (infoMatch) {
         sysInfo[infoMatch[1]] = infoMatch[2];
      }
    } else if (currentSection && line.trim().length > 0 && !line.startsWith('╔') && !line.startsWith('║') && !line.startsWith('╚') && !line.startsWith('=')) {
      // It's likely an instruction line like: "  add    z0.b, z1.b, z2.b               (INT8 add)           ok"
      const parts = line.split(/[()]+/); // splits by ( or )
      if (parts.length >= 1) {
         // The instruction is the part before the first (
         const parsedLine = line.trim();
         let status = 'unknown';
         if (parsedLine.toLowerCase().endsWith('ok')) status = 'ok';
         else if (parsedLine.toLowerCase().endsWith('sigill')) status = 'sigill';

         if (status !== 'unknown') {
             // Find where the description ( ... ) starts and ends
             const openParen = line.indexOf('(');
             const closeParen = line.indexOf(')');
             
             let instruction = line;
             let description = '';
             if (openParen !== -1 && closeParen !== -1) {
                 instruction = line.substring(0, openParen).trim();
                 description = line.substring(openParen + 1, closeParen).trim();
             } else {
                 instruction = parsedLine.substring(0, parsedLine.lastIndexOf(' '));
             }

             // If it's a bracketed status like [ ✅ OK ], it might be different, but cleanText has ok or sigill at end
             currentSection.items.push({
                 instruction,
                 description,
                 status
             });
         }
      }
    }
  }
  
  const dataDir = join(__dirname, '../src/data');
  if (!existsSync(dataDir)) {
     mkdirSync(dataDir, { recursive: true });
  }
  
  const finalData = { 
    timestamp: new Date().toISOString(), 
    sysInfo, 
    sections: sections.filter(s => s.items.length > 0) 
  };
  writeFileSync(join(dataDir, 'probe_results.json'), JSON.stringify(finalData, null, 2));
  console.log('Successfully updated probe_results.json with real hardware data.');
} catch (e) {
  console.error('Failed to run probe or parse output:', e);
}
