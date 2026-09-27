import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import EntryIndex from "@/components/entry-index";
import { getEntrySummaries } from "@/lib/content";

export const metadata: Metadata = {
  title: "Studies",
  description:
    "Technical notes and experiments in online learning, NLP, attention, and sequence models by Yong-Hwan Lee.",
  alternates: { canonical: "/studies/" },
};

export default function Studies() {
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
          Learning in public
        </p>
        <h1 className="text-4xl font-semibold tracking-tighter">
          Studies & notes
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
          Working through algorithms with code, derivations, and experiments.
        </p>
      </header>
      <EntryIndex entries={getEntrySummaries("studies")} section="studies" />
    </main>
  );
}
