import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import EntryIndex from "@/components/entry-index";
import { getEntrySummaries } from "@/lib/content";

export const metadata: Metadata = {
  title: "Projects",
  description:
    "Projects in AI agents, multi-agent orchestration, conversational CAD, LLM fine-tuning, reinforcement learning, and statistical estimation by Yong-Hwan Lee.",
  alternates: { canonical: "/projects/" },
};

export default function Projects() {
  return (
    <main id="main" className="space-y-8">
      <Link
        href="/"
        className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3.5" aria-hidden="true" /> Home
      </Link>
      <header>
        <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          Research · Experiments · Systems
        </p>
        <h1 className="text-4xl font-semibold tracking-tighter">Projects</h1>
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
          The problem, the approach, and what I learned along the way.
        </p>
      </header>
      <EntryIndex entries={getEntrySummaries("projects")} section="projects" />
    </main>
  );
}
