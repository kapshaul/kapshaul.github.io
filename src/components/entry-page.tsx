import Link from "next/link";
import { ArrowLeft, ArrowRight, ArrowUpRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { MarkdownContent } from "@/components/markdown-content";
import ProjectVideo from "@/components/project-video";
import { projectDisplay } from "@/data/projects";
import { getEntries, type Entry } from "@/lib/content";
import { formatDate } from "@/lib/utils";

export default function EntryPage({ entry }: { entry: Entry }) {
  const display =
    entry.section === "projects" ? projectDisplay[entry.slug] : undefined;
  const entries = getEntries(entry.section);
  const index = entries.findIndex((item) => item.slug === entry.slug);
  const next = entries[(index + 1) % entries.length];
  // The downloads are presented as buttons in the article header instead.
  const body = entry.body
    .replace(
      /^#{1,6}\s+Download\s*\r?\n[\s\S]*?(?=^---\s*$|^#{1,6}\s|$(?![\s\S]))/m,
      "",
    )
    .replace(/^(?:\s*---\s*\r?\n)+/, "");

  return (
    <main id="main">
      <Link
        href={`/${entry.section}/`}
        className="mb-9 inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3.5" aria-hidden="true" /> All{" "}
        {entry.section}
      </Link>
      <article>
        <header className="space-y-5 border-b pb-8">
          <div className="flex flex-wrap items-center gap-3 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground">
            <span>
              {entry.section === "projects"
                ? display?.category || "Project"
                : entry.category || "Study notes"}
            </span>
            {display?.status && (
              <span className="rounded-full border px-2 py-0.5">
                {display.status}
              </span>
            )}
          </div>
          <h1 className="text-3xl font-semibold leading-[1.15] tracking-tighter sm:text-4xl">
            {display?.title || entry.title}
          </h1>
          <p className="text-base leading-relaxed text-muted-foreground">
            {display?.summary || entry.summary}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {(display?.tags || entry.tags).map((tag) => (
              <Badge
                variant="secondary"
                key={tag}
                className="rounded-md px-2.5 py-1 text-[10px] font-normal"
              >
                {tag}
              </Badge>
            ))}
          </div>
          <div className="space-y-1 text-xs leading-relaxed text-muted-foreground">
            <p>{entry.authors.join(" · ")}</p>
            {display?.period ? (
              <p>Period · {display.period}</p>
            ) : entry.date && (
              <p>
                Published{" "}
                <time dateTime={entry.date}>{formatDate(entry.date)}</time>
                {entry.updated && entry.updated !== entry.date && (
                  <>
                    {" "}
                    · Updated{" "}
                    <time dateTime={entry.updated}>
                      {formatDate(entry.updated)}
                    </time>
                  </>
                )}
              </p>
            )}
          </div>
          {entry.links.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-1">
              {entry.links.map((link) => (
                <a
                  key={link.url}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium hover:bg-muted"
                >
                  {link.label}
                  <ArrowUpRight className="size-3" aria-hidden="true" />
                </a>
              ))}
            </div>
          )}
        </header>
        <div className="py-8">
          {display?.video && (
            <figure className="mb-8">
              <div className="overflow-hidden rounded-xl border">
                <ProjectVideo video={display.video} />
              </div>
              <figcaption className="mt-3 text-xs leading-relaxed text-muted-foreground">
                {display.video.caption}
              </figcaption>
            </figure>
          )}
          <MarkdownContent
            content={body}
            basePath={`/${entry.section}/${entry.slug}/`}
            showTableOfContents={entry.section === "studies"}
          />
        </div>
      </article>
      {next && next.slug !== entry.slug && (
        <Link
          href={`/${entry.section}/${next.slug}/`}
          className="group mt-8 flex items-center justify-between gap-6 rounded-xl border p-5 transition-colors hover:bg-muted/50"
        >
          <div>
            <p className="mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
              Keep exploring
            </p>
            <p className="text-sm font-medium">
              {entry.section === "projects"
                ? projectDisplay[next.slug]?.title || next.title
                : next.title}
            </p>
          </div>
          <ArrowRight
            className="size-5 shrink-0 transition-transform group-hover:translate-x-1"
            aria-hidden="true"
          />
        </Link>
      )}
    </main>
  );
}
