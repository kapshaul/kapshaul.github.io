import Image from "next/image";
import Link from "next/link";
import { ArrowUpRight, FileText, Github } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { Entry } from "@/lib/content";
import { projectDisplay } from "@/data/projects";
import ProjectVideo from "@/components/project-video";

export default function ProjectCard({ entry }: { entry: Omit<Entry, "body"> }) {
  const display = projectDisplay[entry.slug];
  const isJarvis = entry.slug === "jarvis";
  const href = `/projects/${entry.slug}/`;
  const code = entry.links.find((link) =>
    link.url.startsWith("https://github.com/"),
  );
  const documentLink = entry.links.find((link) => /\.pdf(?:$|[?#])/i.test(link.url));

  return (
    <article className={`group flex flex-col overflow-hidden rounded-xl border border-border bg-card transition-[box-shadow,border-color] duration-200 hover:border-foreground/25 hover:shadow-md hover:shadow-black/5 ${isJarvis ? "min-[520px]:col-span-2" : ""}`}>
      {display?.video ? (
        <div className="overflow-hidden border-b">
          <ProjectVideo video={display.video} />
        </div>
      ) : (
        <Link
          href={href}
          tabIndex={-1}
          aria-hidden="true"
          className="relative block aspect-[16/10] max-h-56 overflow-hidden border-b bg-white"
        >
          {entry.image ? (
            <Image
              src={entry.image}
              alt=""
              fill
              sizes="(max-width: 640px) 100vw, 320px"
              className="object-contain p-5 transition-transform duration-500 group-hover:scale-[1.035]"
            />
          ) : (
            <div className="flex h-full items-center justify-center font-mono text-sm text-neutral-500">
              {display?.category || "AI & Systems"}
            </div>
          )}
          {display?.status && (
            <span className="absolute left-3 top-3 rounded-full border border-neutral-200 bg-white px-2.5 py-1 text-[10px] font-medium text-neutral-600">
              {display.status}
            </span>
          )}
        </Link>
      )}
      <div className={`flex flex-col gap-3 p-5 ${isJarvis ? "sm:p-6" : ""}`}>
        <p className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
          {display?.category || "Project"}
          {entry.date ? ` · ${entry.date.slice(0, 4)}` : ""}
          {display?.video && display.status ? ` · ${display.status}` : ""}
        </p>
        <h3 className={`text-base font-semibold leading-snug tracking-tight ${isJarvis ? "sm:text-xl" : ""}`}>
          <Link href={href} className="hover:underline underline-offset-4">
            {display?.title || entry.title}
          </Link>
        </h3>
        <p className="text-[13px] leading-relaxed text-muted-foreground">
          {display?.summary || entry.summary}
        </p>
        {display?.highlight && (
          <p className="border-l-2 border-foreground/25 pl-2.5 text-xs leading-relaxed">
            {display.highlight}
          </p>
        )}
        <div className="flex flex-wrap gap-1.5 pt-1">
          {(display?.tags || entry.tags).map((tag) => (
            <Badge
              key={tag}
              variant="secondary"
              className="rounded-md px-2 py-0.5 text-[10px] font-normal"
            >
              {tag}
            </Badge>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t pt-3 text-xs font-medium">
          <Link href={href} className="inline-flex items-center gap-1">
            Case study <ArrowUpRight className="size-3" aria-hidden="true" />
          </Link>
          {code && (
            <a
              href={code.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"
            >
              <Github className="size-3" aria-hidden="true" /> Code
            </a>
          )}
          {documentLink && (
            <a
              href={documentLink.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"
            >
              <FileText className="size-3" aria-hidden="true" /> Document
            </a>
          )}
        </div>
      </div>
    </article>
  );
}
