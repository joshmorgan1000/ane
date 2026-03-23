import React, { useState } from 'react';
import { Cpu, MemoryStick, Layers, Code2, AlertTriangle, MonitorPlay, Zap, Table as TableIcon, Baseline, BookOpen, Search, X, Info } from 'lucide-react';
import probeData from './data/probe_results.json';
import throughputData from './data/throughput_results.json';
import mnistData from './data/mnist_results.json';
import armDocsData from './data/arm_docs.json';

const chip = probeData?.sysInfo?.chip || 'M-Series';

export default function App() {
  const [activeTab, setActiveTab] = useState<'mnist' | 'registers' | 'throughput' | 'overview' | 'operations'>('mnist');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-sans selection:bg-cyan-900 selection:text-cyan-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/50 sticky top-0 z-10 backdrop-blur-sm shrink-0">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-4">
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
        
        <nav className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 flex space-x-6 overflow-x-auto">
          {[
            { id: 'mnist', icon: Baseline, label: 'MNIST Benchmarks' },
            { id: 'registers', icon: MemoryStick, label: 'Registers & Memory' },
            { id: 'throughput', icon: Zap, label: 'Throughput (TOPS)' },
            { id: 'overview', icon: MonitorPlay, label: 'Nuances & Overview' },
            { id: 'operations', icon: Code2, label: 'ARM Index' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 py-4 -mb-[1px] border-b-2 text-sm font-medium transition-colors whitespace-nowrap ${
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

      <main className={`flex-1 max-w-[1600px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 ${activeTab === 'operations' ? 'flex flex-col' : 'space-y-12'}`}>
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'registers' && <RegistersTab />}
        {activeTab === 'operations' && <CombinedOperationsTab />}
        {activeTab === 'throughput' && <ThroughputTab />}
        {activeTab === 'mnist' && <MnistBenchmarksTab />}
      </main>
    </div>
  );
}

function CombinedOperationsTab() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedOp, setSelectedOp] = useState<any>(null);

  const probeItems = probeData.sections.filter((sec: any) => sec.items && sec.items[0] && !sec.items[0].label).flatMap((sec: any) => 
    sec.items.map((item: any) => ({ ...item, category: sec.name.replace(/\[.*\]/, "").trim() }))
  );

  const filteredItems = probeItems.filter((item: any) => 
    item.instruction.toLowerCase().includes(searchQuery.toLowerCase()) || 
    (item.description && item.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const getDoc = (instr: string) => {
    const mnemonic = instr.split(' ')[0].toUpperCase().replace(/[^A-Z0-9]/g, '');
    return (armDocsData as any)[mnemonic];
  };

  return (
    <div className="animate-in fade-in flex flex-col lg:flex-row gap-6 h-[calc(100vh-14rem)] w-full relative">
      {/* MASTER PANE */}
      <div className="w-full lg:w-1/3 xl:w-1/4 flex flex-col bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl shrink-0 h-[40vh] lg:h-auto">
        <div className="p-4 border-b border-slate-800 bg-slate-900/80 sticky top-0 z-10 shrink-0">
          <div className="flex items-center space-x-2 text-slate-200 font-bold mb-3">
            <Code2 className="w-5 h-5 text-cyan-400" />
            <h3>Operations ({filteredItems.length})</h3>
          </div>
          <div className="relative group w-full">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-slate-500 group-focus-within:text-cyan-400 transition-colors" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="block w-full pl-9 pr-8 py-2 text-sm border border-slate-700/80 rounded-lg bg-slate-950 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 focus:border-cyan-500 transition-all font-mono"
              placeholder="Search mnemonics..."
            />
            {searchQuery && (
              <button onClick={() => setSearchQuery('')} className="absolute inset-y-0 right-0 pr-2 flex items-center text-slate-500 hover:text-slate-300">
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto divide-y divide-slate-800/50 bg-slate-900/30 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
          {filteredItems.length === 0 ? (
            <div className="p-6 text-center text-slate-500">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-20" />
              <p className="text-sm">No operations found</p>
            </div>
          ) : (
            filteredItems.map((item: any, i: number) => {
              const isSelected = selectedOp?.instruction === item.instruction;
              const hasDoc = !!getDoc(item.instruction);
              
              return (
                <button 
                  key={i}
                  onClick={() => setSelectedOp(item)}
                  className={`w-full text-left px-4 py-3 transition-colors hover:bg-slate-800/50 focus:outline-none flex flex-col group ${
                    isSelected ? 'bg-slate-800/80 border-l-2 border-cyan-500' : 'border-l-2 border-transparent'
                  }`}
                >
                  <div className="flex justify-between items-start mb-1 w-full relative">
                    <code className={`font-mono text-sm leading-tight pr-4 ${isSelected ? 'text-cyan-400 font-bold' : 'text-slate-300 group-hover:text-cyan-300'}`}>
                      {item.instruction.split(' ')[0]}
                    </code>
                    <div className="flex items-center space-x-2 shrink-0">
                       {hasDoc && <BookOpen className={`w-3.5 h-3.5 ${isSelected ? 'text-fuchsia-400' : 'text-slate-600 group-hover:text-cyan-400'}`} />}
                       <span className={`w-2 h-2 rounded-full ${item.status === 'ok' ? 'bg-emerald-500' : item.status === 'compile_fail' ? 'bg-yellow-500' : 'bg-red-500'}`}></span>
                    </div>
                  </div>
                  <div className="text-xs text-slate-500 truncate w-full">{item.instruction}</div>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* DETAIL PANE */}
      <div className="w-full lg:w-2/3 xl:w-3/4 flex flex-col bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl flex-1 relative">
        {selectedOp ? (
          <div className="flex flex-col h-full">
            {/* Detail Header */}
            <div className="shrink-0 p-6 md:p-8 border-b border-slate-800 bg-slate-900/50">
              <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-4">
                <div>
                  <h2 className="text-2xl md:text-3xl font-bold text-cyan-400 font-mono mb-2">{selectedOp.instruction}</h2>
                  <p className="text-slate-400 text-lg">{selectedOp.description}</p>
                </div>
                <div className="flex flex-col items-end shrink-0">
                  <span className={`px-4 py-1.5 rounded-full font-bold uppercase tracking-wider border text-sm flex items-center shadow-inner ${selectedOp.status === 'ok' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
                    <span className={`w-2 h-2 rounded-full mr-2 ${selectedOp.status === 'ok' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
                    {selectedOp.status === 'ok' ? 'ACTIVE CHIP SUPPORT' : (selectedOp.status || 'TRAP').toUpperCase()}
                  </span>
                  <span className="text-xs text-slate-500 mt-3 font-mono">
                    Category: {selectedOp.category}
                  </span>
                </div>
              </div>
            </div>

            {/* Scrollable Doc Content */}
            <div className="flex-1 overflow-y-auto p-6 md:p-8 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
              <h3 className="text-xl font-bold text-slate-200 mb-6 flex items-center sticky top-0 bg-slate-900 py-2 z-10">
                <BookOpen className="w-6 h-6 mr-3 text-fuchsia-400" />
                ARM Architecture Reference
              </h3>
              
              {(() => {
                const docData = getDoc(selectedOp.instruction);
                return docData && docData.length > 0 ? (
                  <div className="space-y-6 pb-6">
                    {docData.map((doc: string, i: number) => (
                      <div key={i} className="bg-slate-950 p-5 md:p-6 rounded-lg text-slate-300 font-mono text-sm leading-relaxed border border-slate-800/80 whitespace-pre-wrap shadow-inner overflow-x-auto">
                        {doc}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-slate-950/50 p-12 rounded-xl text-center border border-slate-800/80 border-dashed">
                    <Info className="w-12 h-12 mx-auto mb-4 text-slate-600 opacity-50" />
                    <p className="text-slate-400 font-medium text-lg">No exact documentation mapped for this mnemonic.</p>
                    <p className="text-slate-600 text-sm mt-2">This may be an alias or a highly specific variation not individually outlined in the base reference text.</p>
                  </div>
                );
              })()}
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-slate-500 bg-slate-900/30">
            <BookOpen className="w-16 h-16 mb-4 opacity-20" />
            <p className="text-xl font-medium">Select an operation to view details</p>
            <p className="text-sm mt-2 opacity-60">Browse the list or search for a mnemonic to explore native ARM docs</p>
          </div>
        )}
      </div>
    </div>
  );
}

function MnistBenchmarksTab() {
  const formatSPS = (val: number) => {
    return new Intl.NumberFormat('en-US').format(Math.round(val));
  };
  
  const speedup = mnistData.pytorch.throughput > 0 
    ? (mnistData.sme.throughput / mnistData.pytorch.throughput).toFixed(1)
    : 'N/A';

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <div className="flex items-center space-x-3 mb-6">
          <Baseline className="w-6 h-6 text-fuchsia-400" />
          <h2 className="text-2xl font-bold text-slate-100">MNIST Training Benchmarks</h2>
        </div>
        
        <p className="text-slate-300 max-w-4xl mb-8 leading-relaxed">
          This benchmark evaluates a simple 3-layer neural network (784 → 128 → 10) on the MNIST dataset. It compares the custom bare-metal SME C++ implementation executing hand-tuned SME bytecode against PyTorch CPU (Eager). Performance is measured in samples evaluated per second.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-bl-full blur-3xl"></div>
            <h3 className="text-xl font-bold text-slate-100 mb-2 truncate">SME Native (C++)</h3>
            <div className="flex items-end space-x-2">
              <span className="text-4xl font-bold text-cyan-400">{formatSPS(mnistData.sme.throughput)}</span>
              <span className="text-slate-400 pb-1">samples/sec</span>
            </div>
            <div className="mt-4 flex items-center justify-between text-sm">
              <span className="text-slate-400">Final Accuracy</span>
              <span className="text-emerald-400 font-mono">{mnistData.sme.accuracy}%</span>
            </div>
          </div>
          
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-orange-500/10 rounded-bl-full blur-3xl"></div>
            <h3 className="text-xl font-bold text-slate-100 mb-2 truncate">PyTorch CPU (Eager)</h3>
            <div className="flex items-end space-x-2">
              <span className="text-4xl font-bold text-orange-400">{formatSPS(mnistData.pytorch.throughput)}</span>
              <span className="text-slate-400 pb-1">samples/sec</span>
            </div>
            <div className="mt-4 flex items-center justify-between text-sm">
              <span className="text-slate-400">Final Accuracy</span>
              <span className="text-emerald-400 font-mono">{mnistData.pytorch.accuracy}%</span>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-fuchsia-500/10 to-transparent border-l-4 border-fuchsia-500 p-6 rounded-r-xl">
          <h3 className="text-lg font-bold text-slate-100 mb-2">Performance Analysis</h3>
          <p className="text-slate-300">
            The SME Native implementation achieves a <strong className="text-fuchsia-400">{speedup}x speedup</strong> over PyTorch CPU. This demonstrates the immense raw throughput available when bypassing heavy framework overhead and directly targeting Apple Silicon's Matrix Coprocessor via hand-tuned micro-ops.
          </p>
        </div>
      </section>
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <h2 className="text-2xl font-bold text-slate-100 mb-4">{chip} Hardware Nuances</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card title="SME2 vs SME1 Parsing" icon={<Code2 className="text-emerald-400" />}>
            <p>
              The {chip} uses the newer SME2 instruction format structure. Standard 8-bit multiple-vector outer products (which LLMs often expect as <code>smopa</code>) use the <strong className="text-emerald-300">sdot</strong> instruction with multiple vectors instead. For instance, <code className="text-emerald-200 bg-slate-800 px-1 rounded">sdot za.s[w0, 0, vgx4], {'{z0.b-z3.b}'}, z4.b</code> essentially performs a 4-way SMOPA.
            </p>
          </Card>

          <Card title="No FP8 Support" icon={<AlertTriangle className="text-amber-400" />}>
            <p>
              Unlike some other modern architectures, the Apple <strong className="text-amber-300">{chip} does not support FP8 natively</strong> (missing <code>FEAT_SME_F8</code>). All FP8 operations (such as <code>fmopa</code> expecting FP8) trap via <code>SIGILL</code> (Illegal Instruction). Only operations on Int8, BF16, FP16, FP32, and FP64 succeed.
            </p>
          </Card>

          <Card title="8-Bit MOPA Verification" icon={<Zap className="text-cyan-400" />}>
            <p>
              {chip} fully implements complete combinations of signs via <strong className="text-cyan-300">smopa, umopa, sumopa, and usmopa</strong> across all standard 32-bit output tiles (<code>za0.s</code> - <code>za3.s</code>).
            </p>
          </Card>
        </div>
      </section>
      
      <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center"><Cpu className="w-5 h-5 mr-2 text-indigo-400" /> LLM & Compiler Hallucinations</h3>
        <p className="text-slate-400 leading-relaxed">
          Because Apple's architecture is historically undocumented regarding its exact internal matrix engine layouts, documentation bots and LLMs consistently claim 8-bit operations skip certain tiles, or that SME2 instructions do not exist. On the {chip}, they are highly present. For example, raw assembly requires exact binary <code>.inst</code> (hex payloads) inside <code>clang</code> because standard LLVM assemblers may aggressively reject valid matrix operands that the silicon fully supports.
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
              On the Apple {chip}, the Vector Length (VL) is <strong className="text-cyan-300">128 bits (16 bytes)</strong> during standard SVE execution.
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
                <span>4 Byte tiles (8-bit elements)</span>
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
        <section>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 h-full">
            <h3 className="text-lg font-bold text-slate-200 mb-2">The ZA Aliasing Model</h3>
            <p className="text-sm text-slate-400 mb-4 leading-relaxed">
              The predefined ZA tiles (Byte, Halfword, Word, Doubleword) are <strong>not separate chunks of memory</strong>. They are overlapping architectural aliases matching the same flat 4KB block. Modifying <code>ZA0.B</code> implicitly modifies interleaved segments of <code>ZA0.S</code>, <code>ZA1.S</code>, etc.
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
        </section>
        
        <section>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 h-full flex flex-col">
            <h3 className="text-lg font-bold text-slate-200 mb-2">Vertical vs Horizontal Slices</h3>
            <p className="text-sm text-slate-400 leading-relaxed mb-6">
              Instead of loading whole matrices, SVE mandates chunking memory into <strong>1D Vector Slices</strong> horizontally or vertically.
            </p>
            
            <div className="space-y-4 flex-grow">
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
        </section>
      </div>

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

function ThroughputTab() {
  const throughputSections = probeData.sections.filter((sec: any) => sec.items[0] && !!sec.items[0].label);
  
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section>
        <h2 className="text-2xl font-bold text-slate-100 flex items-center mb-6">
          <Zap className="w-6 h-6 mr-3 text-yellow-400" />
          Hardware Throughput Matrix (TOPS)
        </h2>
        <div className="text-slate-400 mb-6 bg-slate-900 border border-slate-800 p-5 rounded-xl">
          <p>These benchmarks represent <strong>measured, sustained operations per second (TOPS)</strong> by deploying multi-threaded isolated kernels onto the {chip} structure simultaneously. They prove that the SME unit operates entirely independently of the GPU and BNNS hardware, reinforcing the theory that it behaves similarly to the core Apple Neural Engine hardware matrix pathways.</p>
          <div className="mt-3 text-xs text-slate-500 font-mono flex items-center">
            <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse mr-2"></span>
            Executed at: {new Date(throughputSections.length > 0 ? probeData.timestamp : throughputData.timestamp).toLocaleString()}
          </div>
        </div>

        {throughputSections.length > 0 && (
          <div className="space-y-6 mt-8">
            {throughputSections.map((sec: any, idx: number) => (
              <div key={idx} className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden mt-6">
                <div className="bg-slate-900 px-5 py-3 border-b border-slate-800">
                  <h4 className="font-semibold text-slate-300 text-sm">{sec.name.replace(/\[.*\]/, "").trim()}</h4>
                </div>
                <div className="divide-y divide-slate-800/60 bg-slate-950">
                  {sec.items.map((item: any, i: number) => (
                    <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between p-4 hover:bg-slate-900/40 transition-colors">
                      <div className="text-slate-200 font-mono text-sm mb-2 sm:mb-0 max-w-lg">{item.label}</div>
                      <div className="flex items-center gap-4">
                        <div className="text-emerald-400 font-mono font-bold text-lg">{item.value.toFixed(3)} <span className="text-xs text-slate-500 font-normal ml-1">{item.unit}</span></div>
                        <div className="text-slate-500 font-mono text-xs w-20 text-right">{item.timeInfo}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
        
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl mt-8">
          <div className="px-5 py-4 border-b border-slate-800">
             <h4 className="font-semibold text-slate-300">Subsystem Overlap Tests</h4>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs uppercase bg-slate-950 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4 font-semibold">Test Configuration</th>
                  <th className="px-6 py-4 font-semibold text-right">BNNS INT8</th>
                  <th className="px-6 py-4 font-semibold text-right">CBLAS FP32</th>
                  <th className="px-6 py-4 font-semibold text-right">GPU FP16</th>
                  <th className="px-6 py-4 font-semibold text-right">NEON FP32</th>
                  <th className="px-6 py-4 font-semibold text-right">SME</th>
                  <th className="px-6 py-4 font-semibold text-right text-cyan-400">TOTAL TOPS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {throughputData.results.map((row: any, idx: number) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-6 py-4 font-medium text-slate-200">{row.test}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.bnns > 0 ? row.bnns.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.cblas > 0 ? row.cblas.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.gpu > 0 ? row.gpu.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.neon > 0 ? row.neon.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-slate-400">{row.sme > 0 ? row.sme.toFixed(3) : '--'}</td>
                    <td className="px-6 py-4 text-right font-mono text-cyan-400 font-bold">{row.total.toFixed(3)}</td>
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
