"use client";

import { Evaluation } from "@/lib/api";
import { cn } from "@/lib/utils";
import { CheckCircle, XCircle, AlertTriangle, Brain, Target, Mic, Code, MessageSquare } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { SkillTags } from "./SkillTags";
import { CandidateRadarChart } from "./charts/RadarChart";

const DECISION_CONFIG = {
  SELECT: {
    label: "SELECTED",
    color: "#10b981",
    bg: "rgba(16, 185, 129, 0.1)",
    border: "#10b981",
    icon: CheckCircle,
    badgeClass: "badge-select",
  },
  CONSIDER: {
    label: "CONSIDER",
    color: "#f59e0b",
    bg: "rgba(245, 158, 11, 0.1)",
    border: "#f59e0b",
    icon: AlertTriangle,
    badgeClass: "badge-consider",
  },
  REJECT: {
    label: "REJECTED",
    color: "#ef4444",
    bg: "rgba(239, 68, 68, 0.1)",
    border: "#ef4444",
    icon: XCircle,
    badgeClass: "badge-reject",
  },
};

function scoreColor(score: number) {
  if (score >= 70) return "#10b981";
  if (score >= 50) return "#f59e0b";
  return "#ef4444";
}

interface EvaluationResultProps {
  evaluation: Evaluation;
  candidateName?: string;
}

export function EvaluationResult({ evaluation, candidateName }: EvaluationResultProps) {
  const decision = (evaluation.final_decision || "REJECT").toUpperCase() as keyof typeof DECISION_CONFIG;
  const config = DECISION_CONFIG[decision] || DECISION_CONFIG.REJECT;
  const DecisionIcon = config.icon;

  const metrics = [
    {
      label: "Match Score",
      value: evaluation.match_score ?? 0,
      display: `${(evaluation.match_score ?? 0).toFixed(1)}%`,
      color: scoreColor(evaluation.match_score ?? 0),
      icon: Target,
    },
    {
      label: "Skill Ratio",
      value: (evaluation.skill_match_ratio ?? 0) * 100,
      display: `${((evaluation.skill_match_ratio ?? 0) * 100).toFixed(0)}%`,
      color: scoreColor((evaluation.skill_match_ratio ?? 0) * 100),
      icon: Code,
    },
    {
      label: "Comm Score",
      value: (evaluation.comm_score ?? 0) * 100,
      display: (evaluation.comm_score ?? 0).toFixed(2),
      color: "#00d4ff",
      icon: Mic,
    },
    {
      label: "ML Confidence",
      value: evaluation.ml_confidence ?? 0,
      display: `${(evaluation.ml_confidence ?? 0).toFixed(1)}%`,
      color: "#7c3aed",
      icon: Brain,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Decision Banner */}
      <div
        className="rounded-xl p-6 border-l-4"
        style={{ background: config.bg, borderColor: config.border, borderLeftWidth: "4px" }}
      >
        <div className="flex items-start justify-between">
          <div>
            {candidateName && (
              <div className="font-mono text-xl font-bold text-white mb-1">{candidateName}</div>
            )}
            <div className="flex items-center gap-2">
              <DecisionIcon className="h-5 w-5" style={{ color: config.color }} />
              <span className="font-mono text-sm" style={{ color: config.color }}>
                {config.label}
              </span>
            </div>
          </div>
          <div className="text-right">
            <div
              className="font-mono text-4xl font-bold"
              style={{ color: config.color }}
            >
              {(evaluation.match_score ?? 0).toFixed(1)}%
            </div>
            <div className="text-[#64748b] text-xs uppercase tracking-widest">Match Score</div>
            {evaluation.model_used && (
              <div className="text-[#64748b] text-xs mt-1">
                via {evaluation.model_used.replace("_", " ")}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((m) => (
          <div key={m.label} className="card-cyber p-4 text-center">
            <m.icon className="h-4 w-4 mx-auto mb-2" style={{ color: m.color }} />
            <div className="font-mono text-2xl font-bold mb-1" style={{ color: m.color }}>
              {m.display}
            </div>
            <Progress value={m.value} className="h-1 mb-1" />
            <div className="text-[#64748b] text-xs uppercase tracking-wider">{m.label}</div>
          </div>
        ))}
      </div>

      {/* Skills Breakdown */}
      <div className="grid lg:grid-cols-2 gap-4">
        <div className="card-cyber p-5">
          <div className="section-label mb-3">Skills Analysis</div>
          <div className="space-y-2 text-sm text-[#94a3b8] mb-3">
            <div className="flex justify-between">
              <span>Matched Skills</span>
              <span className="text-[#10b981] font-mono font-bold">
                {evaluation.num_matched_skills ?? 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Missing Skills</span>
              <span className="text-[#ef4444] font-mono font-bold">
                {evaluation.num_missing_skills ?? 0}
              </span>
            </div>
          </div>
          <SkillTags
            matched={evaluation.matched_skills || []}
            missing={evaluation.missing_skills || []}
          />
        </div>

        <div className="card-cyber p-5">
          <div className="section-label mb-3">Interview Analysis</div>
          <div className="space-y-3 text-sm">
            {[
              { label: "Tech Depth", value: (evaluation.tech_score ?? 0) * 100, color: "#00d4ff" },
              { label: "Sentiment", value: ((evaluation.conf_score ?? 0) + 1) / 2 * 100, color: "#7c3aed" },
              { label: "Vocabulary", value: (evaluation.comm_score ?? 0) * 100, color: "#10b981" },
            ].map((item) => (
              <div key={item.label}>
                <div className="flex justify-between mb-1 text-[#94a3b8]">
                  <span>{item.label}</span>
                  <span className="font-mono" style={{ color: item.color }}>
                    {item.value.toFixed(0)}%
                  </span>
                </div>
                <Progress value={Math.min(item.value, 100)} className="h-1.5" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Radar Chart */}
      {evaluation.match_score !== undefined && (
        <div className="card-cyber p-5">
          <div className="section-label mb-3">Candidate Profile Radar</div>
          <CandidateRadarChart evaluation={evaluation} />
        </div>
      )}

      {/* Explanation */}
      {evaluation.explanation && (
        <div
          className="rounded-lg p-4 border"
          style={{
            background: "rgba(0,212,255,0.05)",
            borderColor: "rgba(0,212,255,0.2)",
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <MessageSquare className="h-4 w-4 text-[#00d4ff]" />
            <span className="section-label">AI Explanation</span>
          </div>
          <p className="text-[#94a3b8] text-sm leading-relaxed">{evaluation.explanation}</p>
        </div>
      )}
    </div>
  );
}
