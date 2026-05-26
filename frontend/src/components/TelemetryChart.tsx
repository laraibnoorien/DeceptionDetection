import React from 'react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid 
} from 'recharts';

interface TelemetryChartProps {
  data: any[];
  dataKey: string;
  color: string;
  height?: number;
  domain?: [number, number];
}

export const TelemetryChart: React.FC<TelemetryChartProps> = ({ 
  data, 
  dataKey, 
  color, 
  height = 60,
  domain = [0, 1]
}) => {
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
          <Line 
            type="monotone" 
            dataKey={dataKey} 
            stroke={color} 
            strokeWidth={1.5} 
            dot={false}
            animationDuration={300}
          />
          <YAxis hide domain={domain} />
          <XAxis hide />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-slate-900 border border-slate-800 p-1.5 rounded-md shadow-xl">
                    <p className="text-[10px] font-mono text-slate-300">
                      {payload[0].value?.toLocaleString()}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
