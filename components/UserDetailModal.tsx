import React, { useState, useMemo } from 'react';
import { User, Tweet, BotDetectionResult } from '../types';
import { X, Bot, User as UserIcon } from 'lucide-react';
import { detectBot } from '../features';

interface UserDetailModalProps {
  user: User;
  tweets: Tweet[];
  onClose: () => void;
  existingResult?: BotDetectionResult;
  onAnalyzeComplete: (result: BotDetectionResult) => void;
  groundTruthBot?: boolean;
  pythonScore?: number;
}

// Score bar component for displaying feature scores
const ScoreBar = ({ label, score, highlight }: { label: string; score: number; highlight?: boolean }) => {
  const percentage = Math.min(score * 100, 100);
  const color = score > 0.7 ? 'bg-red-500' : score > 0.4 ? 'bg-orange-500' : 'bg-emerald-500';

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className={highlight ? 'text-white font-semibold' : 'text-slate-400'}>{label}</span>
        <span className={highlight ? 'text-white font-semibold' : 'text-slate-300'}>{(score * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export const UserDetailModal: React.FC<UserDetailModalProps> = ({
  user,
  tweets,
  onClose,
  existingResult,
  groundTruthBot,
  pythonScore
}) => {
  const [result] = useState<BotDetectionResult | undefined>(existingResult);

  // Generate detailed analysis for display
  const detailedAnalysis = useMemo(() => {
    if (tweets.length === 0) return null;
    return detectBot(user, tweets, pythonScore);
  }, [user, tweets, pythonScore]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 w-full max-w-4xl max-h-[90vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        
        {/* Header */}
        <div className="p-6 border-b border-slate-700 flex justify-between items-start bg-slate-800/50">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-400 text-2xl font-bold border border-indigo-500/30">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                {user.name}
                <span className="text-slate-400 text-base font-normal">@{user.username}</span>
              </h2>
              <div className="flex items-center gap-3 mt-1 text-sm text-slate-400">
                <span>ID: {user.id}</span>
                {groundTruthBot !== undefined && (
                   <span className={`px-2 py-0.5 rounded text-xs font-bold ${groundTruthBot ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                     {groundTruthBot ? 'KNOWN BOT' : 'KNOWN HUMAN'}
                   </span>
                )}
              </div>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-700 rounded-full transition-colors text-slate-400 hover:text-white">
            <X size={24} />
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Left Column: Stats & AI */}
          <div className="w-1/3 p-6 border-r border-slate-700 overflow-y-auto bg-slate-800/20">
            <div className="space-y-6">
              
              {/* Profile Info */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Profile</h3>
                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700/50 space-y-2">
                  <p className="text-sm text-slate-300"><span className="text-slate-500 block text-xs">Description</span> {user.description || <em className="text-slate-600">No description</em>}</p>
                  <p className="text-sm text-slate-300"><span className="text-slate-500 block text-xs">Location</span> {user.location || <em className="text-slate-600">Unknown</em>}</p>
                  <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-slate-700">
                     <div>
                       <span className="text-xs text-slate-500">Tweets</span>
                       <p className="text-lg font-mono text-white">{user.tweet_count}</p>
                     </div>
                     <div>
                       <span className="text-xs text-slate-500">Z-Score</span>
                       <p className={`text-lg font-mono ${user.z_score && user.z_score > 3 ? 'text-red-400' : 'text-emerald-400'}`}>
                         {user.z_score?.toFixed(2) ?? 'N/A'}
                       </p>
                     </div>
                  </div>
                </div>
              </div>

              {/* Analysis */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex justify-between items-center">
                  Analysis
                </h3>

                {result ? (
                   <div className={`p-4 rounded-xl border ${result.isBot ? 'bg-red-500/10 border-red-500/50' : 'bg-emerald-500/10 border-emerald-500/50'}`}>
                      <div className="flex items-center gap-3 mb-2">
                        {result.isBot ? <Bot className="text-red-400" size={24} /> : <UserIcon className="text-emerald-400" size={24} />}
                        <div>
                          <p className={`font-bold text-lg ${result.isBot ? 'text-red-400' : 'text-emerald-400'}`}>
                            {result.isBot ? 'Likely Bot' : 'Likely Human'}
                          </p>
                          <p className="text-xs text-slate-400">Confidence: {(result.confidence * 100).toFixed(0)}%</p>
                        </div>
                      </div>
                      <p className="text-sm text-slate-300 italic">"{result.reasoning}"</p>
                   </div>
                ) : (
                  <div className="text-center p-6 border border-dashed border-slate-700 rounded-xl text-slate-500 text-sm">
                      Run heuristic scan from the main dashboard to see analysis.
                  </div>
                )}
              </div>

              {/* Feature Breakdown */}
              {detailedAnalysis && (
                <div className="space-y-3">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Feature Scores</h3>
                  <div className="bg-slate-800 p-3 rounded-xl border border-slate-700/50 space-y-2">
                    <ScoreBar label="Z-Score" score={detailedAnalysis.scores.zScore} />
                    <ScoreBar label="Temporal" score={detailedAnalysis.scores.temporal} />
                    <ScoreBar label="Content" score={detailedAnalysis.scores.content} />
                    <ScoreBar label="Linguistic" score={detailedAnalysis.scores.linguistic} />
                    <ScoreBar label="Profile" score={detailedAnalysis.scores.profile} />
                    {pythonScore !== undefined && (
                      <ScoreBar label="Python RF Model" score={detailedAnalysis.scores.pythonModel} />
                    )}
                    <div className="pt-2 mt-2 border-t border-slate-700">
                      <ScoreBar label="Final Score" score={detailedAnalysis.scores.final} highlight />
                    </div>
                  </div>
                </div>
              )}

            </div>
          </div>

          {/* Right Column: Tweet Feed */}
          <div className="w-2/3 flex flex-col bg-slate-900">
             <div className="p-4 bg-slate-800/30 border-b border-slate-700">
               <h3 className="text-sm font-semibold text-white">Recent Activity ({tweets.length})</h3>
             </div>
             <div className="overflow-y-auto p-4 space-y-4">
               {tweets.map(tweet => (
                 <div key={tweet.id} className="bg-slate-800 p-4 rounded-lg border border-slate-700/50 hover:border-slate-600 transition-colors">
                   <p className="text-slate-200 whitespace-pre-wrap">{tweet.text}</p>
                   <div className="flex justify-between items-center mt-3 pt-3 border-t border-slate-700/50 text-xs text-slate-500">
                     <span>{new Date(tweet.created_at).toLocaleString()}</span>
                     <span className="bg-slate-700 px-2 py-0.5 rounded text-slate-300">{tweet.lang}</span>
                   </div>
                 </div>
               ))}
               {tweets.length === 0 && (
                 <div className="text-center py-10 text-slate-500">
                   No tweets found for this user in the dataset window.
                 </div>
               )}
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};