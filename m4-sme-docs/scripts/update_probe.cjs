const { execSync } = require('child_process');
const { writeFileSync, mkdirSync, existsSync } = require('fs');
const { join } = require('path');

try {
  console.log('Running hardware probe script to fetch live data... (this executes actual objdump and Apple clang, compiling C payloads)');
  
  const probePath = join(__dirname, '../../probes/probe_instructions.sh');
  // Run the bash script and capture output, ignoring stderr exceptions
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
      const parsedLine = line.trim();
      // Parse status and optional error details
      // Formats: "ok", "SIGILL|exit=132|sig=4", "COMPILE_FAIL|error detail"
      let status = 'unknown';
      let errorDetail = '';
      let exitCode = null;
      let signalNum = null;

      if (parsedLine.toLowerCase().endsWith('ok')) {
        status = 'ok';
      } else {
        // Check for pipe-delimited error info
        const parts = parsedLine.split('|');
        const lastWord = parts[0].trim().split(/\s+/).pop().toLowerCase();
        if (lastWord === 'sigill' || lastWord === 'sigsegv' || lastWord === 'sigbus' ||
            lastWord === 'sigtrap' || lastWord === 'sigabrt' || lastWord === 'unknown') {
          status = lastWord;
          for (const part of parts.slice(1)) {
            const kv = part.trim();
            if (kv.startsWith('exit=')) exitCode = parseInt(kv.slice(5));
            else if (kv.startsWith('sig=')) signalNum = parseInt(kv.slice(4));
            else errorDetail = kv;
          }
        } else if (parsedLine.includes('COMPILE_FAIL')) {
          status = 'compile_fail';
          const pipeIdx = parsedLine.indexOf('|');
          if (pipeIdx !== -1) errorDetail = parsedLine.substring(pipeIdx + 1).trim();
        }
      }

      if (status !== 'unknown') {
          const openParen = line.indexOf('(');
          const closeParen = line.indexOf(')');

          let instruction = line;
          let description = '';
          if (openParen !== -1 && closeParen !== -1) {
              instruction = line.substring(0, openParen).trim();
              description = line.substring(openParen + 1, closeParen).trim();
          } else {
              // Strip everything after the status marker
              instruction = parts[0].trim();
              instruction = instruction.substring(0, instruction.lastIndexOf(' ')).trim();
          }

          const item = { instruction, description, status };
          if (errorDetail) item.error = errorDetail;
          if (exitCode !== null) item.exitCode = exitCode;
          if (signalNum !== null) item.signal = signalNum;

          currentSection.items.push(item);
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
  console.log('Successfully updated probe_results.json with real LIVE hardware data!');
} catch (e) {
  console.error('Failed to run probe or parse output:', e);
}
