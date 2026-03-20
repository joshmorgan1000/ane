const fs = require('fs');
const path = require('path');

const text = fs.readFileSync(path.join(__dirname, '../../clean_probe.txt'), 'utf-8');
const sections = [];
let currentSection = null;
const sysInfo = {};

const lines = text.split('\n');
for (const line of lines) {
  if (line.includes('━━━')) {
    const match = line.match(/━━━\s*(.*?)\s*━━━/);
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
    let status = 'unknown';
    if (parsedLine.toLowerCase().endsWith('ok')) status = 'ok';
    else if (parsedLine.toLowerCase().endsWith('sigill')) status = 'sigill';

    if (status !== 'unknown') {
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

        currentSection.items.push({
            instruction,
            description,
            status
        });
    }
  }
}

const finalData = { 
  timestamp: new Date().toISOString(), 
  sysInfo, 
  sections: sections.filter(s => s.items.length > 0) 
};

// Check if data directory exists, otherwise create it
const dataDir = path.join(__dirname, '../src/data');
if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
}

fs.writeFileSync(path.join(dataDir, 'probe_results.json'), JSON.stringify(finalData, null, 2));
console.log('Written data! Total sections: ' + finalData.sections.length);
