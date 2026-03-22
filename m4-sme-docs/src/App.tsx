import React, { useState } from 'react';
import { Cpu, MemoryStick, Layers, Code2, AlertTriangle, MonitorPlay, Zap, Table as TableIcon, Activity } from 'lucide-react';
import probeData from './data/probe_results.json';
import throughputData from './data/throughput_results.json';

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'registers' | 'operations' | 'memory' | 'probe' | 'throughput'>('overview');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-sans selection:bg-cyan-900 selection:text-cyan-100">
      <header className="border-b border-slate-800 bg-slate-900/50 sticky top-0 z-10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3 text-cyan-400">
              <Zap className="w-8 h-8" />
              <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Apple Silicon SME Architecture Docs</h1>
            </div>
            <div className="hidden sm:flex items-center px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-pulse"></span>
              Live Hardware Data Loaded
            </div>
          </div>
          <p className="mt-1 text-sm text-slate-400">Definitive hardware nuances & undocumented features (SVE / SME2)</p>
        </div>
        
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-6 overflow-x-auto">
          {[
            { id: 'overview', icon: MonitorPlay, label: 'Nuances & Overview' },
            { id: 'registers', icon: Layers, label: 'Register Sizes & Slices' },
            { id: 'operations', icon: Code2, label: 'Operations & Mnemonics' },
            { id: 'memory', icon: MemoryStick, label: 'Memory Overlaps' },
            { id: 'probe', icon: Activity, label: 'Live Probe Results' },
            { id: 'throughput', icon: Zap, label: 'Throughput (TOPS)' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 py-4 border-b-2 text-sm font-medium transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? 'border-cyan-500 text-cyan-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'registers' && <RegistersTab />}
        {activeTab === 'operations' && <OperationsTab />}
        {activeTab === 'memory' && <MemoryTab />}
        {activeTab === 'probe' && <ProbeDataTab />}
        {activeTab === 'throughput' && <ThroughputTab />}
      </main>
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <h2 className="text-2xl font-bold text-slate-100 mb-4">M4 Hardware Nuances</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card title="Undocumented Byte Tiles" icon={<Layers className="text-purple-400" />}>
            <p>
              By default, ARM specifications suggest only <code>ZA0.B</code> exists for 8-bit operations. However, hardware probing confirms the Apple M4 silicon supports <strong className="text-purple-300">ZA1.B, ZA2.B, and ZA3.B</strong>.
              These can be accessed normally using encodings such as <code className="bg-slate-800 px-1 rounded">0xc0000000</code> to <code className="bg-slate-800 px-1 rounded">0xc0000003</code>.
            </p>
          </Card>
          
          <Card title="SME2 vs SME1 Parsing" icon={<Code2 className="text-emerald-400" />}>
            <p>
              The M4 uses the newer SME2 instruction format structure. Standard 8-bit multiple-vector outer products (which LLMs often expect as <code>smopa</code>) use the <strong className="text-emerald-300">sdot</strong> instruction with multiple vectors instead. For instance, <code className="text-emerald-200 bg-slate-800 px-1 rounded">sdot za.s[w0, 0, vgx4], {'{z0.b-z3.b}'}, z4.b</code> essentially performs a 4-way SMOPA.
            </p>
          </Card>

          <Card title="No FP8 Support" icon={<AlertTriangle className="text-amber-400" />}>
            <p>
              Unlike some other modern architectures, the Apple <strong className="text-amber-300">M4 does not support FP8 natively</strong> (missing <code>FEAT_SME_F8</code>). All FP8 operations (such as <code>fmopa</code> expecting FP8) trap via <code>SIGILL</code> (Illegal Instruction). Only operations on Int8, BF16, FP16, FP32, and FP64 succeed.
            </p>
          </Card>

          <Card title="8-Bit MOPA Verification" icon={<Zap className="text-cyan-400" />}>
            <p>
              M4 fully implements complete combinations of signs via <strong className="text-cyan-300">smopa, umopa, sumopa, and usmopa</strong> across all standard 32-bit output tiles (<code>za0.s</code> - <code>za3.s</code>). 
            </p>
          </Card>
        </div>
      </section>
      
      <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center"><Cpu className="w-5 h-5 mr-2 text-indigo-400" /> LLM & Compiler Hallucinations</h3>
        <p className="text-slate-400 leading-relaxed">
          Because Apple's architecture is historically undocumented regarding its exact internal matrix engine layouts, documentation bots and LLMs consistently claim 8-bit operations skip certain tiles, or that SME2 instructions do not exist. On the M4, they are highly present. For example, raw assembly requires exact binary <code>.inst</code> (hex payloads) inside <code>clang</code> because standard LLVM assemblers may aggressively reject valid matrix operands that the silicon fully supports.
        </p>
      </section>
    </div>
  );
}

function RegistersTab() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section>
          <h2 className="text-2xl font-bold text-slate-100 flex items-center mb-6">
            <Layers className="w-6 h-6 mr-3 text-cyan-400" />
            The Z Registers (SVE/SME Vectors)
          </h2>
          <div className="space-y-4 text-slate-400 leading-relaxed bg-slate-900 p-6 rounded-xl border border-slate-800 h-full">
            <p>
              There are <strong className="text-slate-200">32 distinct vector registers: Z0 through Z31</strong>.
            </p>
            <p>
              On the Apple M4, the Vector Length (VL) is <strong className="text-cyan-300">128 bits (16 bytes)</strong> during standard SVE execution.
            </p>
            <p>
              When the CPU enters Streaming SVE mode (SME), the Streaming Vector Length (SVL) is extended to <strong className="text-cyan-300">512 bits (64 bytes)</strong>, providing massive throughput per vector register.
            </p>
            <div className="p-4 bg-slate-950 rounded-lg mt-4 border border-slate-800">
              <h4 className="text-sm font-semibold text-slate-300 mb-2">Multi-vector Groupings (SME2)</h4>
              <p className="text-sm mb-3">SME2 introduces the concept of operating on sequences of vectors. You can target:</p>
              <ul className="list-disc list-inside mt-2 space-y-2 text-sm text-slate-300">
                <li><strong className="text-emerald-400 font-mono">{"{Z0.B-Z1.B}"}</strong> — <span className="text-slate-500">2-Way (Two adjacent vector registers)</span></li>
                <li><strong className="text-emerald-400 font-mono">{"{Z0.B-Z3.B}"}</strong> — <span className="text-slate-500">4-Way (Four adjacent vector registers)</span></li>
              </ul>
              <p className="text-xs text-slate-500 mt-4 border-t border-slate-800 pt-3">
                Registers must usually be aligned to multiples of 2 or 4 when using these groupings (e.g. starting at Z0, Z4, Z8).
              </p>
            </div>
          </div>
        </section>

        <section>
          <h2 className="text-2xl font-bold text-slate-100 flex items-center mb-6">
            <TableIcon className="w-6 h-6 mr-3 text-purple-400" />
            The ZA Array (Matrix Storage)
          </h2>
          <div className="space-y-4 text-slate-400 leading-relaxed bg-slate-900 p-6 rounded-xl border border-slate-800 h-full">
            <p>
              The <strong>ZA array</strong> is a massive 2D matrix structure. For SVL = 512, ZA holds <code>512 x 512 bits</code>, equating to <strong className="text-purple-300">4096 bytes (4 KB)</strong> of architectural state.
            </p>
            
            <p>
              Instead of addressing the whole array or individual cells, it is mathematically layered into <strong className="text-slate-200">Tiles</strong> overlaid onto each other explicitly based on the datatype width.
            </p>

            <ul className="space-y-3 mt-4 text-sm bg-slate-950 p-4 rounded-lg border border-slate-800">
              <li className="flex justify-between items-center border-b border-slate-800 pb-2">
                <span className="font-semibold text-slate-300 font-mono">ZA0.B - ZA3.B</span>
                <span className="text-right flex flex-col">
                  <span>4 Byte tiles (8-bit elements)</span>
                  <span className="text-[10px] text-purple-400 uppercase tracking-widest mt-1 font-bold">M4 Undocumented Exclusives</span>
                </span>
              </li>
              <li className="flex justify-between items-center border-b border-slate-800 pb-2">
                <span className="font-semibold text-slate-300 font-mono">ZA0.H - ZA1.H</span>
                <span>2 Halfword tiles (16-bit elements)</span>
              </li>
              <li className="flex justify-between items-center border-b border-slate-800 pb-2">
                <span className="font-semibold text-slate-300 font-mono">ZA0.S - ZA3.S</span>
                <span>4 Word tiles (32-bit elements)</span>
              </li>
              <li className="flex justify-between items-center">
                <span className="font-semibold text-slate-300 font-mono">ZA0.D - ZA7.D</span>
                <span>8 Doubleword tiles (64-bit elements)</span>
              </li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}

function OperationsTab() {
  const operations = [
    {
      name: "sdot za.s[w0, 0, vgx4], {z0.b-z3.b}, z4.b",
      type: "SME2 Matrix",
      desc: "4-way Signed Dot Product (Outer Product). Performs 8-bit outer product multiplication between elements of vector sequences. It generates a 32-bit accumulative update into a specific ZA tile, treating the input vectors as matrices. This is how M4 natively handles multi-way outer products instead of legacy single-tile SMOPAs.",
      code: "0xa0812000 // Multi-vector SDOT"
    },
    {
      name: "smopa za0.s, p0/m, p0/m, z0.b, z1.b",
      type: "SME1 Outer Product",
      desc: "Signed Matrix Outer Product and Accumulate. Takes two 8-bit Z vectors (columns and rows), multiplies them, inflates the result to 32 bits, and accumulates directly into the `za0.s` (32-bit Word) tile. Predicates (p0) govern active lines.",
      code: "0xa0010000 // Standard SMOPA"
    },
    {
      name: "mova za1h.b[w0, 0], p0/m, z0.b",
      type: "Data Movement",
      desc: "Move vector to ZA Array slice. Moves horizontal (h) or vertical (v) slices of data from Z registers directly into the specified ZA tile byte index. M4 natively supports undoc tiles za1-3.b.",
      code: "0xc0000001 // Write to ZA1.B"
    },
    {
      name: "usmopa za0.s, p0/m, p0/m, z0.b, z1.b",
      type: "Cross-Sign Outer",
      desc: "Unsigned / Signed Matrix Outer Product. Similar to smopa, but treats the left vector as Unsigned 8-bit and the right vector as Signed 8-bit before expanding to 32-bit logic. Crucial for asymmetric quantization schemes (e.g. U8 weights, S8 activations).",
      code: "0xa0410000 // USMOPA"
    },
    {
      name: "zero {za}",
      type: "Clear State",
      desc: "Completely zeroes out the entire 4KB ZA Matrix array. Typically executed explicitly right before the beginning of macro-tile operations mapped out for an attention block or feedforward layer to prevent garbage data accumulation.",
      code: "zero {za}"
    }
  ];

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <h2 className="text-2xl font-bold text-slate-100 mb-6">Mnemonics & Core Operations</h2>
      
      <div className="space-y-4">
        {operations.map((op, i) => (
          <div key={i} className="bg-slate-900 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-colors">
            <div className="flex flex-col md:flex-row md:items-start justify-between">
              <div className="flex-1 mr-6">
                <h3 className="font-mono text-lg font-bold text-cyan-400">{op.name}</h3>
                <span className="inline-block mt-2 px-2 py-0.5 rounded text-xs font-medium bg-indigo-900/50 text-indigo-300 border border-indigo-700/50">
                  {op.type}
                </span>
                <p className="mt-3 text-slate-400 text-sm leading-relaxed">
                  {op.desc}
                </p>
              </div>
              <div className="mt-4 md:mt-0 shrink-0 bg-black/50 p-3 rounded-lg border border-slate-800 h-fit">
                <code className="text-xs font-mono text-emerald-400 whitespace-pre">{op.code}</code>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MemoryTab() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <h2 className="text-2xl font-bold text-slate-100 mb-4 flex items-center">
          <MemoryStick className="w-6 h-6 mr-3 text-indigo-400" />
          Memory Overlaps & Slice Alignment
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-slate-200 mb-2">The ZA Aliasing Model</h3>
            <p className="text-sm text-slate-400 mb-4 leading-relaxed">
              The predefined ZA tiles (Byte, Halfword, Word, Doubleword) are <strong>not separate chunks of memory</strong>. They are overlapping architectural aliases matching the same flat 4KB block. 
              Modifying <code>ZA0.B</code> implicitly modifies interleaved segments of <code>ZA0.S</code>, <code>ZA1.S</code>, etc.
            </p>
            <div className="h-48 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-center relative overflow-hidden my-4 group">
               <div className="grid grid-cols-4 grid-rows-4 w-full h-full opacity-10 divide-x divide-y divide-slate-700 border-slate-500 absolute inset-0">
                 {Array.from({length: 16}).map((_, i) => <div key={i} className="flex items-center justify-center"></div>)}
               </div>
               <div className="absolute inset-0 z-10 p-6 flex flex-col justify-center items-center pointer-events-none">
                 <div className="h-12 w-full bg-purple-500/20 border-2 border-purple-500 rounded flex justify-center items-center mb-2 text-sm font-semibold text-purple-200 shadow-[0_0_15px_rgba(168,85,247,0.3)] transition-transform group-hover:scale-105">4KB ZA Matrix Block</div>
                 <div className="flex w-full justify-between h-12 gap-2 transition-transform group-hover:scale-105">
                   <div className="flex-1 bg-cyan-500/30 border-2 border-cyan-500 rounded flex justify-center items-center text-xs font-medium text-cyan-200">ZA0.S</div>
                   <div className="flex-1 bg-blue-500/30 border-2 border-blue-500 rounded flex justify-center items-center text-xs font-medium text-blue-200">ZA1.S</div>
                   <div className="flex-1 bg-indigo-500/30 border-2 border-indigo-500 rounded flex justify-center items-center text-xs font-medium text-indigo-200">ZA2.S</div>
                   <div className="flex-1 bg-sky-500/30 border-2 border-sky-500 rounded flex justify-center items-center text-xs font-medium text-sky-200">ZA3.S</div>
                 </div>
               </div>
            </div>
            <p className="text-xs text-slate-500 mt-4 text-center">
              * The 32-bit (Word) arrays are 4 interleaved grids mapping strictly to modulo-4 offsets of identical rows.
            </p>
          </div>
          
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-lg font-bold text-slate-200 mb-2">Vertical vs Horizontal Slices</h3>
              <p className="text-sm text-slate-400 leading-relaxed mb-6">
                Instead of loading whole matrices, SVE mandates chunking memory into <strong>1D Vector Slices</strong> horizontally or vertically.
              </p>
              
              <div className="space-y-4">
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                  <h4 className="text-emerald-400 font-mono text-sm mb-1">za.s[w0, 0, vgx4]</h4>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Addresses a set of <strong>Vertical (v)</strong> grouped slices. <code>vgx4</code> denotes a multiple vertical slice group spanning the vertical width of the matrix tiles, fundamental to 4-way operations in SME2.
                  </p>
                </div>
                
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                  <h4 className="text-cyan-400 font-mono text-sm mb-1">za1h.s[w0, 3]</h4>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Targets a single <strong>Horizontal (h)</strong> slice on Word Tile ZA1. It computes the row index using the base scalar register <code>W0</code> with an immediate offset of <code>3</code>.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function Card({ title, icon, children }: { title: string, icon: React.ReactNode, children: React.ReactNode }) {
  return (
    <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-xl relative overflow-hidden group hover:border-slate-700 transition-colors">
      <div className="relative z-10">
        <h3 className="text-lg font-bold text-slate-100 flex items-center mb-3">
          <span className="mr-2">{icon}</span>
          {title}
        </h3>
        <div className="text-sm text-slate-400 leading-relaxed">
          {children}
        </div>
      </div>
    </div>
  );
}

function ProbeDataTab() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <h2 className="text-2xl font-bold text-slate-100 flex items-center mb-6">
          <Activity className="w-6 h-6 mr-3 text-emerald-400" />
          Live Hardware Ground Truth
        </h2>
        <div className="text-slate-400 mb-6 bg-slate-900 border border-slate-800 p-5 rounded-xl">
          <p>This data is not hardcoded. These capabilities were extracted locally by compiling raw hexadecimal opcode payloads and tracking hardware traps. The output proves the exact capabilities of this silicon. Errors show the actual signal received (SIGILL, SIGSEGV, SIGBUS) and exit codes.</p>
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
              <div className={`font-mono text-lg font-bold ${value === "1" ? "text-emerald-400" : "text-slate-600"}`}>{String(value)}</div>
            </div>
          ))}
        </div>

        <h3 className="text-lg font-bold text-slate-200 mb-4">Hardware Capability Matrix</h3>
        <div className="space-y-6">
          {probeData.sections.map((sec, idx) => (
            <div key={idx} className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
              <div className="bg-slate-900 px-5 py-3 border-b border-slate-800">
                <h4 className="font-semibold text-slate-300 text-sm">{sec.name.replace(/\[.*\]/, "").trim()}</h4>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-slate-800">
                {sec.items.map((item: any, i: number) => {
                  const statusConfig = item.status === "ok"
                    ? { bg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20", label: "SUPPORTED" }
                    : item.status === "compile_fail"
                    ? { bg: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20", label: "COMPILE FAIL" }
                    : item.status === "sigsegv"
                    ? { bg: "bg-orange-500/10 text-orange-400 border-orange-500/20", label: "SIGSEGV" }
                    : item.status === "sigbus"
                    ? { bg: "bg-orange-500/10 text-orange-400 border-orange-500/20", label: "SIGBUS" }
                    : { bg: "bg-red-500/10 text-red-400 border-red-500/20", label: item.status === "sigill" ? "SIGILL" : (item.status || "TRAP").toUpperCase() };
                  return (
                  <div key={i} className="bg-slate-950 p-4 flex flex-col justify-center">
                    <div className="flex justify-between items-start mb-2">
                       <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider shrink-0 border ${statusConfig.bg}`}>
                         {statusConfig.label}
                       </span>
                       {item.signal != null && item.status !== "ok" && (
                         <span className="text-[9px] font-mono text-slate-600 ml-2">
                           exit={item.exitCode} sig={item.signal}
                         </span>
                       )}
                    </div>
                    <code className="block font-mono text-sm text-cyan-300 mb-1 whitespace-pre-wrap">{item.instruction}</code>
                    {item.description && <div className="text-[11px] text-slate-500">{item.description}</div>}
                    {item.error && <div className="text-[10px] text-red-400/60 font-mono mt-1 truncate">{item.error}</div>}
                  </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ThroughputTab() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <h2 className="text-2xl font-bold text-slate-100 flex items-center mb-6">
          <Zap className="w-6 h-6 mr-3 text-yellow-400" />
          Hardware Throughput Matrix (TOPS)
        </h2>
        <div className="text-slate-400 mb-6 bg-slate-900 border border-slate-800 p-5 rounded-xl">
          <p>These benchmarks represent <strong>measured, sustained operations per second (TOPS)</strong> by deploying multi-threaded isolated kernels onto the M4 structure simultaneously. They prove that the SME unit operates entirely independently of the GPU and BNNS hardware, reinforcing the theory that it behaves similarly to the core Apple Neural Engine hardware matrix pathways.</p>
          <div className="mt-3 text-xs text-slate-500 font-mono flex items-center">
            <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse mr-2"></span>
            Executed at: {new Date(throughputData.timestamp).toLocaleString()}
          </div>
        </div>
        
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl mt-8">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs uppercase bg-slate-950 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-semibold">Concurrency Strategy</th>
                  <th className="px-6 py-4 font-semibold text-right">BNNS</th>
                  <th className="px-6 py-4 font-semibold text-right">GPU</th>
                  <th className="px-6 py-4 font-semibold text-right">NEON</th>
                  <th className="px-6 py-4 font-semibold text-right">SME</th>
                  <th className="px-6 py-4 font-semibold text-right text-cyan-400">TOTAL TOPS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {throughputData.results.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-6 py-4 font-medium text-slate-200">{row.test}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.bnns > 0 ? row.bnns.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.gpu > 0 ? row.gpu.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.neon > 0 ? row.neon.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400 font-semibold">{row.sme > 0 ? row.sme.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-cyan-400 font-bold">{row.total > 0 ? row.total.toFixed(3) : '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
