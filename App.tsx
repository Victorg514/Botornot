import React, { useState, useMemo } from 'react';
import { Upload, FileJson, AlertCircle, CheckCircle, Search, Filter, Play, Download, RefreshCw } from 'lucide-react';
import { Dataset, User, Tweet, BotDetectionResult, ScoringMetrics } from './types';
import { StatsCard } from './components/StatsCard';
import { UserDetailModal } from './components/UserDetailModal';
import { detectBot, optimizeEnsembleWeights, EnsembleWeights } from './features';
import { exportBotDetections, calculateStats, exportDetailedReport } from './exportResults';

// Enhanced analysis using heuristic + optional python ensemble
const runAnalysis = (
  users: User[],
  allTweets: Tweet[],
  pythonScores: Map<string, number>,
  ensembleWeights?: EnsembleWeights,
): BotDetectionResult[] => {
  const hasPython = pythonScores.size > 0;
  return users.map(user => {
    const userTweets = allTweets.filter(t => t.author_id === user.id);
    const pyScore = pythonScores.get(user.id);
    const detection = detectBot(user, userTweets, pyScore, ensembleWeights);

    return {
      userId: user.id,
      isBot: detection.isBot,
      confidence: detection.confidence,
      reasoning: detection.reasoning,
      method: hasPython ? 'ensemble' as const : 'heuristic' as const,
    };
  });
};

function App() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [groundTruth, setGroundTruth] = useState<Set<string>>(new Set());
  const [analysisResults, setAnalysisResults] = useState<Map<string, BotDetectionResult>>(new Map());
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [filter, setFilter] = useState<'all' | 'bots' | 'humans'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [pythonScores, setPythonScores] = useState<Map<string, number>>(new Map());
  const [ensembleWeights, setEnsembleWeights] = useState<EnsembleWeights | undefined>(undefined);
  const [optimizeProgress, setOptimizeProgress] = useState('');
  const [heuristicResults, setHeuristicResults] = useState<Map<string, BotDetectionResult>>(new Map());
  const [ensembleResults, setEnsembleResults] = useState<Map<string, BotDetectionResult>>(new Map());
  const [useEnsemble, setUseEnsemble] = useState(true);

  const handleReset = () => {
    if (window.confirm("Are you sure? This will clear current data and results.")) {
      setDataset(null);
      setGroundTruth(new Set());
      setAnalysisResults(new Map());
      setSelectedUser(null);
      setFilter('all');
      setSearchTerm('');
      setIsProcessing(false);
      setPythonScores(new Map());
      setEnsembleWeights(undefined);
      setOptimizeProgress('');
      setHeuristicResults(new Map());
      setEnsembleResults(new Map());
      setUseEnsemble(true);
    }
  };

  // File Upload Handlers
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target?.result as string);
        // Basic validation based on prompt structure
        if (json.posts && json.users) {
          setDataset(json);
          // Clear previous state
          setAnalysisResults(new Map());
          setGroundTruth(new Set());
        } else {
          alert("Invalid dataset format. Expected { posts: [], users: [], ... }");
        }
      } catch (err) {
        alert("Error parsing JSON file");
      }
    };
    reader.readAsText(file);
  };

  const handleGroundTruthUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const ids = text.split('\n').map(id => id.trim()).filter(id => id.length > 0);
      setGroundTruth(new Set(ids));
    };
    reader.readAsText(file);
  };

  // Main scan: auto-fetch python scores, optimize if possible, then run
  const runBatchAnalysis = async () => {
    if (!dataset) return;
    setIsProcessing(true);
    setOptimizeProgress('');

    // Step 1: Auto-fetch python_scores.json if not already loaded
    let scores = pythonScores;
    if (scores.size === 0) {
      try {
        setOptimizeProgress('Fetching python_scores.json...');
        const res = await fetch('/python_scores.json');
        if (res.ok) {
          const json = await res.json();
          const loaded = new Map<string, number>();
          for (const [userId, prob] of Object.entries(json)) {
            if (typeof prob === 'number') loaded.set(userId, prob);
          }
          scores = loaded;
          setPythonScores(loaded);
          setOptimizeProgress(`Loaded ${loaded.size} python scores`);
        }
      } catch {
        // No python scores available — run heuristic only
      }
    }

    // Check if python scores actually cover this dataset's users
    if (scores.size > 0 && dataset) {
      const matchCount = dataset.users.filter(u => scores.has(u.id)).length;
      const coverage = matchCount / dataset.users.length;
      if (coverage < 0.5) {
        // Python scores don't match this dataset — ignore them
        scores = new Map();
        setOptimizeProgress(`Python scores only cover ${matchCount}/${dataset.users.length} users — using heuristic only`);
      }
    }

    // Step 2: Get ensemble weights
    let weights = ensembleWeights;
    if (scores.size > 0 && !weights) {
      if (groundTruth.size > 0) {
        // Have ground truth → optimize live
        setOptimizeProgress('Optimizing ensemble weights (500k iterations)...');
        await new Promise(r => setTimeout(r, 50));

        const result = optimizeEnsembleWeights(
          dataset.users, dataset.posts, scores, groundTruth, 500000,
        );
        weights = result.weights;
        setEnsembleWeights(weights);
        setOptimizeProgress(
          `Optimized! Score: ${result.score} | W_py: ${weights.weightPython.toFixed(4)} | W_h: ${weights.weightHeuristic.toFixed(4)} | Thresh: ${weights.threshold.toFixed(4)}`
        );

        // Save golden weights for future runs without ground truth
        localStorage.setItem('goldenWeights', JSON.stringify(weights));

        // Auto-download weights.json so user can put it in public/
        const blob = new Blob([JSON.stringify(weights, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'weights.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } else {
        // No ground truth → load saved golden weights from file or localStorage
        try {
          const res = await fetch('/weights.json');
          if (res.ok) {
            weights = await res.json() as EnsembleWeights;
            setEnsembleWeights(weights);
            setOptimizeProgress(
              `Using saved weights | W_py: ${weights.weightPython.toFixed(4)} | W_h: ${weights.weightHeuristic.toFixed(4)} | Thresh: ${weights.threshold.toFixed(4)}`
            );
          }
        } catch {
          // Fallback to localStorage
          const saved = localStorage.getItem('goldenWeights');
          if (saved) {
            try {
              weights = JSON.parse(saved) as EnsembleWeights;
              setEnsembleWeights(weights);
              setOptimizeProgress(
                `Using saved weights | W_py: ${weights.weightPython.toFixed(4)} | W_h: ${weights.weightHeuristic.toFixed(4)} | Thresh: ${weights.threshold.toFixed(4)}`
              );
            } catch { /* ignore bad data */ }
          }
        }
      }
    }

    // Step 3: Always run heuristic-only as a safe baseline
    const heuristicOnly = runAnalysis(dataset.users, dataset.posts, new Map());
    const heuristicMap = new Map<string, BotDetectionResult>();
    heuristicOnly.forEach((r: BotDetectionResult) => heuristicMap.set(r.userId, r));
    setHeuristicResults(heuristicMap);

    // Step 4: Run ensemble if python scores are available
    if (scores.size > 0) {
      const ensResults = runAnalysis(dataset.users, dataset.posts, scores, weights);
      const ensembleMap = new Map<string, BotDetectionResult>();
      ensResults.forEach((r: BotDetectionResult) => ensembleMap.set(r.userId, r));
      setEnsembleResults(ensembleMap);
      setAnalysisResults(ensembleMap);
      setUseEnsemble(true);
    } else {
      setAnalysisResults(heuristicMap);
      setUseEnsemble(false);
    }
    setIsProcessing(false);
  };

  // Metrics Calculation
  const metrics: ScoringMetrics | null = useMemo(() => {
    if (groundTruth.size === 0 || analysisResults.size === 0) return null;

    let tp = 0; // Bot detected as Bot
    let fn = 0; // Bot detected as Human (or missed)
    let fp = 0; // Human detected as Bot

    // Iterate over all known bots from ground truth
    groundTruth.forEach(botId => {
      const result = analysisResults.get(botId);
      if (result?.isBot) {
        tp++;
      } else {
        fn++; // Either marked as human or not analyzed
      }
    });

    // Iterate over our results to find False Positives
    analysisResults.forEach((result: BotDetectionResult, userId: string) => {
      if (result.isBot && !groundTruth.has(userId)) {
        fp++;
      }
    });

    // Score 
    const score = (tp * 4) + (fn * -1) + (fp * -2);

    return { truePositives: tp, falseNegatives: fn, falsePositives: fp, score };
  }, [groundTruth, analysisResults]);

  // Data Filtering
  const filteredUsers = useMemo(() => {
    if (!dataset) return [];
    return dataset.users.filter(u => {
      const matchesSearch = u.username.toLowerCase().includes(searchTerm.toLowerCase()) || 
                            u.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                            u.id.includes(searchTerm);
      
      if (!matchesSearch) return false;

      const result = analysisResults.get(u.id);
      if (filter === 'bots') return result?.isBot;
      if (filter === 'humans') return result && !result.isBot;
      return true;
    });
  }, [dataset, filter, searchTerm, analysisResults]);


  if (!dataset) {
    return (
      <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center p-4">
        <div className="max-w-md w-full bg-slate-800 p-8 rounded-2xl shadow-2xl border border-slate-700 text-center">
          <div className="w-16 h-16 bg-indigo-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
            <Play className="text-indigo-400 fill-indigo-400" size={32} />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">BotOrNot Analyzer</h1>
          <p className="text-slate-400 mb-8">Upload your dataset to begin analysis.</p>
          
          <label className="flex flex-col gap-2 cursor-pointer group">
            <span className="text-sm font-medium text-slate-300 group-hover:text-indigo-400 transition-colors">Upload Dataset JSON</span>
            <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 hover:border-indigo-500 hover:bg-slate-700/30 transition-all">
               <FileJson className="mx-auto text-slate-500 mb-2 group-hover:text-indigo-400" size={32} />
               <span className="text-slate-500 text-sm">Click to browse</span>
            </div>
            <input type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
          </label>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-50 pb-20">
      {/* Navbar */}
      <nav className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-white">B</div>
             <h1 className="font-bold text-xl tracking-tight">BotOrNot <span className="text-indigo-400">Analyzer</span></h1>
          </div>
          <div className="flex items-center gap-4">
            
            {/* 3. Added Scan New Data Button */}
            <button 
              onClick={handleReset}
              className="bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-4 py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 border border-slate-700"
            >
              <RefreshCw size={16} />
              Scan new data
            </button>

            {dataset && (
               <div className="hidden md:flex items-center gap-2 text-sm text-slate-400 bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">
                 <span>{dataset.id}</span>
                 <span className="w-1 h-1 bg-slate-600 rounded-full"></span>
                 <span>{dataset.lang.toUpperCase()}</span>
               </div>
            )}
<label className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2.5 rounded-lg text-sm font-medium cursor-pointer transition-colors flex items-center gap-2">
              <Upload size={16} />
              {groundTruth.size > 0 ? `Ground Truth (${groundTruth.size})` : "Upload Ground Truth"}
              <input type="file" accept=".txt" onChange={handleGroundTruthUpload} className="hidden" />
            </label>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Dashboard Top Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
           <StatsCard title="Total Users" value={dataset.users.length} icon={<Filter size={20} />} color="text-indigo-400" />
           <StatsCard title="Total Posts" value={dataset.posts.length} icon={<FileJson size={20} />} color="text-blue-400" />
           <StatsCard 
             title="Detected Bots" 
             value={Array.from(analysisResults.values()).filter(r => r.isBot).length} 
             icon={<AlertCircle size={20} />} 
             color="text-red-400" 
           />
           <StatsCard 
             title="Score" 
             value={metrics ? metrics.score : 'N/A'} 
             icon={<CheckCircle size={20} />} 
             color={metrics && metrics.score > 0 ? "text-emerald-400" : "text-slate-400"} 
           />
        </div>

        {/* ... (Rest of the component remains exactly the same) ... */}
        
        {/* Action & Metrics Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
           {/* Run Scan & Optimizer */}
           <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg flex flex-col items-center justify-center space-y-4">
             {analysisResults.size === 0 ? (
               <button
                 onClick={runBatchAnalysis}
                 disabled={isProcessing}
                 className="bg-indigo-600 hover:bg-indigo-500 text-white px-8 py-4 rounded-xl text-lg font-medium transition-colors flex items-center gap-3"
               >
                 <Play size={24} />
                 {isProcessing ? "Processing..." : "Run Scan"}
               </button>
             ) : (
               <div className="text-center space-y-2">
                 <CheckCircle size={48} className="text-emerald-400 mx-auto" />
                 <p className="text-lg font-medium text-white">Scan Complete</p>
                 <p className="text-sm text-slate-400">{Array.from(analysisResults.values()).filter(r => r.isBot).length} bots detected out of {analysisResults.size} users</p>

                 {/* Mode toggle — only show if ensemble is available */}
                 {heuristicResults.size > 0 && pythonScores.size > 0 && (
                   <div className="flex gap-2 mt-3">
                     <button
                       onClick={() => { setUseEnsemble(true); setAnalysisResults(ensembleResults); }}
                       className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${useEnsemble ? 'bg-violet-600 text-white' : 'bg-slate-700 text-slate-400 hover:text-white'}`}
                     >
                       Ensemble
                     </button>
                     <button
                       onClick={() => { setUseEnsemble(false); setAnalysisResults(heuristicResults); }}
                       className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${!useEnsemble ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-400 hover:text-white'}`}
                     >
                       Heuristic Only
                     </button>
                   </div>
                 )}

                 {ensembleWeights && useEnsemble && (
                   <div className="mt-2 text-xs text-violet-300 bg-violet-500/10 border border-violet-500/20 rounded-lg p-2">
                     W_python: {ensembleWeights.weightPython.toFixed(4)} | W_heuristic: {ensembleWeights.weightHeuristic.toFixed(4)} | Threshold: {ensembleWeights.threshold.toFixed(4)}
                   </div>
                 )}
               </div>
             )}
             {optimizeProgress && (
               <p className="text-xs text-slate-400 text-center font-mono">{optimizeProgress}</p>
             )}
           </div>

           {/* Score Breakdown */}
           <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
             <h3 className="font-semibold text-lg mb-6">Accuracy Metrics</h3>
             {metrics ? (
               <div className="space-y-4">
                 <div className="flex justify-between items-center p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                   <span className="text-emerald-400 font-medium">True Positives (+4)</span>
                   <span className="text-xl font-bold text-white">{metrics.truePositives}</span>
                 </div>
                 <div className="flex justify-between items-center p-3 bg-red-500/10 rounded-lg border border-red-500/20">
                   <span className="text-red-400 font-medium">False Positives (-2)</span>
                   <span className="text-xl font-bold text-white">{metrics.falsePositives}</span>
                 </div>
                 <div className="flex justify-between items-center p-3 bg-orange-500/10 rounded-lg border border-orange-500/20">
                   <span className="text-orange-400 font-medium">False Negatives (-1)</span>
                   <span className="text-xl font-bold text-white">{metrics.falseNegatives}</span>
                 </div>
                 <div className="mt-6 pt-4 border-t border-slate-700">
                    <div className="flex justify-between items-end">
                      <span className="text-slate-400">Total Score</span>
                      <span className="text-3xl font-bold text-white">{metrics.score}</span>
                    </div>
                 </div>
                 {analysisResults.size > 0 && (
                   <div className="mt-6 pt-4 border-t border-slate-700 space-y-2">
                     <button
                       onClick={() => {
                         const teamName = prompt('Enter your team name:', 'YourTeam');
                         if (teamName) {
                           exportBotDetections(analysisResults, teamName);
                         }
                       }}
                       className="w-full bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                     >
                       <Download size={16} />
                       Export Submission File
                     </button>
                     <button
                       onClick={() => {
                         calculateStats(analysisResults, groundTruth);
                         exportDetailedReport(analysisResults, groundTruth, dataset?.id.toString() || 'dataset');
                       }}
                       className="w-full bg-slate-700 hover:bg-slate-600 text-slate-300 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                     >
                       <FileJson size={16} />
                       Export Analysis Report
                     </button>
                   </div>
                 )}
               </div>
             ) : (
               <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 space-y-2">
                 <Upload size={32} className="opacity-50" />
                 <p>Upload Ground Truth (.txt) file and run a scan to see metrics.</p>
               </div>
             )}
           </div>
        </div>

        {/* User Table */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow-lg overflow-hidden">
          <div className="p-4 border-b border-slate-700 flex flex-col sm:flex-row gap-4 justify-between items-center bg-slate-800/50">
            <div className="flex gap-2">
               <button 
                 onClick={() => setFilter('all')}
                 className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${filter === 'all' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
               >
                 All Users
               </button>
               <button 
                 onClick={() => setFilter('bots')}
                 className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${filter === 'bots' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
               >
                 Detected Bots
               </button>
               <button 
                 onClick={() => setFilter('humans')}
                 className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${filter === 'humans' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
               >
                 Clean
               </button>
            </div>
            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
              <input 
                type="text" 
                placeholder="Search users..." 
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg pl-9 pr-4 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-slate-900/50 text-xs uppercase text-slate-400 font-medium">
                <tr>
                  <th className="px-6 py-4">User</th>
                  <th className="px-6 py-4">Tweets</th>
                  <th className="px-6 py-4">Z-Score</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700">
                {filteredUsers.slice(0, 100).map(user => {
                  const result = analysisResults.get(user.id);
                  const isGroundTruthBot = groundTruth.has(user.id);
                  
                  return (
                    <tr key={user.id} className="hover:bg-slate-700/30 transition-colors">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold text-slate-300">
                            {user.username.charAt(0).toUpperCase()}
                          </div>
                          <div>
                            <div className="font-medium text-white">{user.name}</div>
                            <div className="text-xs text-slate-500">@{user.username}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-slate-300 font-mono">{user.tweet_count}</td>
                      <td className="px-6 py-4">
                        <span className={`font-mono ${user.z_score && user.z_score > 3 ? 'text-orange-400' : 'text-slate-400'}`}>
                          {user.z_score?.toFixed(2) ?? '-'}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        {result ? (
                          <div className="flex flex-col">
                            <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium w-fit ${result.isBot ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'}`}>
                              {result.isBot ? 'BOT' : 'HUMAN'}
                              <span className="ml-1 opacity-75">{(result.confidence * 100).toFixed(0)}%</span>
                            </span>
                          </div>
                        ) : (
                          <span className="text-xs text-slate-500 italic">Unanalyzed</span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button 
                          onClick={() => setSelectedUser(user)}
                          className="text-sm text-indigo-400 hover:text-indigo-300 font-medium"
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  );
                })}
                {filteredUsers.length === 0 && (
                  <tr>
                     <td colSpan={5} className="px-6 py-8 text-center text-slate-500">
                       No users found matching your filters.
                     </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          {filteredUsers.length > 100 && (
             <div className="px-6 py-3 bg-slate-900/50 text-center text-xs text-slate-500 border-t border-slate-700">
               Showing first 100 of {filteredUsers.length} users
             </div>
          )}
        </div>
      </main>

      {/* Modal */}
      {selectedUser && dataset && (
        <UserDetailModal
          user={selectedUser}
          tweets={dataset.posts.filter(p => p.author_id === selectedUser.id)}
          onClose={() => setSelectedUser(null)}
          existingResult={analysisResults.get(selectedUser.id)}
          onAnalyzeComplete={(res) => setAnalysisResults(new Map(analysisResults).set(res.userId, res))}
          groundTruthBot={groundTruth.size > 0 ? groundTruth.has(selectedUser.id) : undefined}
          pythonScore={pythonScores.get(selectedUser.id)}
        />
      )}
    </div>
  );
}

export default App;