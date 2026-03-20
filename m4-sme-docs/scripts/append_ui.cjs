const fs = require('fs');
const current = fs.readFileSync('src/App.tsx', 'utf8');

const additional = `
function ProbeDataTab() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <h2 className="text-2xl font-bold text-slate-100 flex items-center mb-6">
          <Activity className="w-6 h-6 mr-3 text-emerald-400" />
          Live Hardware Ground Truth
        </h2>
        <div className="text-slate-400 mb-6 bg-slate-900 border border-slate-800 p-5 rounded-xl">
          <p>This data is not hardcoded. These capabilities were extracted locally directly from the Apple M4 CPU by compiling raw hexadecimal opcode payloads and tracking hardware traps (\`SIGILL\`). The output proves the exact capabilities of this silicon model.</p>
          <div className="mt-3 text-xs text-slate-500 font-mono flex items-center">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse mr-2"></span>
            Executed at: {new Date(probeData.timestamp).toLocaleString()}
          </div>
        </div>
        
        <h3 className="text-lg font-bold text-slate-200 mb-4 mt-8">System Vector Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-10">
          {Object.entries(probeData.sysInfo).map(([key, value]) => (
            <div key={key} className="bg-slate-900/80 border border-slate-800 rounded-lg p-3 flex flex-col justify-center items-center">
              <div className="text-[10px] text-slate-500 mb-1 tracking-wider uppercase text-center">{key.replace("FEAT_", "")}</div>
              <div className={\`font-mono text-lg font-bold \${value === "1" ? "text-emerald-400" : "text-slate-600"}\`}>{String(value)}</div>
            </div>
          ))}
        </div>

        <h3 className="text-lg font-bold text-slate-200 mb-4">Hardware Capability Matrix</h3>
        <div className="space-y-6">
          {probeData.sections.map((sec, idx) => (
            <div key={idx} className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
              <div className="bg-slate-900 px-5 py-3 border-b border-slate-800">
                <h4 className="font-semibold text-slate-300 text-sm">{sec.name.replace(/\\[.*\\]/, "").trim()}</h4>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-slate-800">
                {sec.items.map((item, i) => (
                  <div key={i} className="bg-slate-950 p-4 flex flex-col justify-center">
                    <div className="flex justify-between items-start mb-2">
                       <span className={\`text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider shrink-0 \${
                         item.status === "ok" 
                           ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" 
                           : "bg-red-500/10 text-red-400 border border-red-500/20"
                       }\`}>
                         {item.status === "ok" ? "SUPPORTED" : "TRAP (SIGILL)"}
                       </span>
                    </div>
                    <code className="block font-mono text-sm text-cyan-300 mb-1 whitespace-pre-wrap">{item.instruction}</code>
                    {item.description && <div className="text-[11px] text-slate-500">{item.description}</div>}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
`;

fs.writeFileSync('src/App.tsx', current + additional);
console.log('Appended successfully');
