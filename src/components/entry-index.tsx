"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowUpRight, Search } from "lucide-react";
import ProjectCard from "@/components/project-card";
import type { Entry } from "@/lib/content";
import { projectDisplay } from "@/data/projects";

export default function EntryIndex({
  entries,
  section,
}: {
  entries: Omit<Entry, "body">[];
  section: "projects" | "studies";
}) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All");
  const categories = [
    "All",
    ...new Set(
      entries
        .map((entry) =>
          section === "projects"
            ? projectDisplay[entry.slug]?.category
            : entry.category,
        )
        .filter((value): value is string => Boolean(value)),
    ),
  ];
  const filtered = entries.filter((entry) => {
    const display = section === "projects" ? projectDisplay[entry.slug] : undefined;
    const entryCategory = display?.category ?? entry.category;
    const searchable = [
      entry.title,
      entry.summary,
      entryCategory,
      ...entry.tags,
      display?.title,
      display?.summary,
      display?.highlight,
      ...(display?.tags || []),
    ]
      .filter(Boolean)
      .join(" ");
    return (
      (category === "All" || entryCategory === category) &&
      searchable.toLowerCase().includes(query.toLowerCase().trim())
    );
  });

  return (
    <div className="space-y-7">
      <div className="relative">
        <Search
          className="absolute left-3 top-3 size-4 text-muted-foreground"
          aria-hidden="true"
        />
        <input
          aria-label={`Search ${section}`}
          type="search"
          placeholder={`Search ${section} by topic or technique…`}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          className="h-10 w-full rounded-lg border bg-background pl-9 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring"
        />
      </div>
      {categories.length > 1 && (
        <div
          role="group"
          aria-label={`Filter ${section}`}
          className="flex flex-wrap gap-2"
        >
          {categories.map((item) => (
            <button
              key={item}
              type="button"
              aria-pressed={category === item}
              onClick={() => setCategory(item)}
              className={`rounded-full border px-3 py-1.5 text-xs transition-colors ${category === item ? "border-foreground bg-foreground text-background" : "text-muted-foreground hover:bg-muted"}`}
            >
              {item}
            </button>
          ))}
        </div>
      )}
      <p
        className="text-xs text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        {filtered.length}{" "}
        {filtered.length === 1
          ? section === "studies"
            ? "study"
            : "project"
          : section}
      </p>
      {filtered.length ? (
        section === "projects" ? (
          <div className="grid items-start gap-4 min-[520px]:grid-cols-2">
            {filtered.map((entry) => (
              <ProjectCard key={entry.slug} entry={entry} />
            ))}
          </div>
        ) : (
          <div className="divide-y">
            {filtered.map((entry) => (
              <Link
                key={entry.slug}
                href={`/studies/${entry.slug}/`}
                className="group block py-6 first:pt-0"
              >
                <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  {[entry.category, entry.date?.slice(0, 4)].filter(Boolean).join(" · ")}
                </p>
                <div className="mb-2 flex items-start justify-between gap-3">
                  <h2 className="font-semibold tracking-tight group-hover:underline underline-offset-4">
                    {entry.title}
                  </h2>
                  <ArrowUpRight
                    className="mt-1 size-4 shrink-0 text-muted-foreground"
                    aria-hidden="true"
                  />
                </div>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {entry.summary}
                </p>
                <p className="mt-3 text-xs text-muted-foreground">
                  {entry.tags.slice(0, 3).join(" / ")}
                </p>
              </Link>
            ))}
          </div>
        )
      ) : (
        <div className="rounded-xl border border-dashed p-10 text-center">
          <p className="font-medium">No {section} found</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Try a different topic or reset the filters.
          </p>
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setCategory("All");
            }}
            className="mt-4 text-sm underline underline-offset-4"
          >
            Reset filters
          </button>
        </div>
      )}
    </div>
  );
}
