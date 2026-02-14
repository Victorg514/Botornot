import React from 'react';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon?: React.ReactNode;
  color?: string; // Tailwind text color class
}

export const StatsCard: React.FC<StatsCardProps> = ({ title, value, icon, color = "text-white" }) => {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 flex items-center justify-between shadow-lg">
      <div>
        <p className="text-slate-400 text-sm font-medium uppercase tracking-wider">{title}</p>
        <h3 className={`text-2xl font-bold mt-1 ${color}`}>{value}</h3>
      </div>
      {icon && <div className={`p-3 rounded-full bg-slate-700/50 ${color}`}>{icon}</div>}
    </div>
  );
};
