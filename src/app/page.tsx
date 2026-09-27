import Image from "next/image";
import Link from "next/link";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Github,
  GraduationCap,
  Mail,
} from "lucide-react";
import BlurFade from "@/components/blur-fade";
import ProjectCard from "@/components/project-card";
import { Badge } from "@/components/ui/badge";
import { profile } from "@/data/profile";
import { featuredSlugs } from "@/data/projects";
import { getEntries } from "@/lib/content";

export default function Home() {
  const projects = getEntries("projects");
  const featured = featuredSlugs.flatMap((slug) =>
    projects.filter((entry) => entry.slug === slug),
  );
  const studies = getEntries("studies");

  return (
    <main id="main" className="flex flex-col gap-16 sm:gap-20">
      <BlurFade>
        <section aria-labelledby="intro" className="space-y-6">
          <div className="flex items-start justify-between gap-5">
            <div className="min-w-0 space-y-3">
              <p className="font-mono text-[10px] font-medium uppercase tracking-[0.2em] text-muted-foreground">
                AI Engineer · Learning & Systems
              </p>
              <h1
                id="intro"
                className="text-[2.35rem] font-semibold leading-[1.1] tracking-[-0.055em] sm:text-5xl"
              >
                Hi, I’m Yong-Hwan
                <span className="text-muted-foreground">.</span>
              </h1>
              <p className="text-sm text-muted-foreground" lang="ko">
                이용환
              </p>
            </div>
            <Image
              src={profile.avatar}
              alt="Yong-Hwan Lee"
              width={128}
              height={128}
              priority
              sizes="(min-width: 640px) 128px, 96px"
              className="size-24 shrink-0 rounded-full border-4 border-background object-cover shadow-sm ring-1 ring-border sm:size-32"
            />
          </div>
          <p className="max-w-[560px] text-base leading-relaxed text-muted-foreground sm:text-lg">
            {profile.description}
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <a
              href="#projects"
              className="inline-flex h-9 items-center gap-2 rounded-lg bg-foreground px-4 text-xs font-medium text-background transition-opacity hover:opacity-80"
            >
              Explore my work{" "}
              <ArrowDown className="size-3.5" aria-hidden="true" />
            </a>
            <a
              href={profile.github}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-9 items-center gap-2 rounded-lg border px-4 text-xs font-medium transition-colors hover:bg-muted"
            >
              <Github className="size-3.5" aria-hidden="true" /> GitHub
            </a>
          </div>
        </section>
      </BlurFade>

      <BlurFade delay={0.08}>
        <section
          id="projects"
          aria-labelledby="projects-heading"
          className="space-y-7 scroll-mt-8"
        >
          <div>
            <div className="mb-3 flex items-center gap-3">
              <Badge className="rounded-md text-[10px] font-medium">
                Selected work
              </Badge>
              <span className="h-px flex-1 bg-border" />
              <span className="font-mono text-[10px] text-muted-foreground">
                01 — {String(featured.length).padStart(2, "0")}
              </span>
            </div>
            <h2
              id="projects-heading"
              className="text-2xl font-semibold tracking-tight sm:text-3xl"
            >
              From research to implementation.
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              A closer look at the models, decisions, and systems behind my
              work.
            </p>
          </div>
          <div className="grid auto-rows-fr gap-4 sm:grid-cols-2">
            {featured.map((entry) => (
              <ProjectCard key={entry.slug} entry={entry} />
            ))}
          </div>
          <Link
            href="/projects/"
            className="group inline-flex items-center gap-2 text-sm font-medium"
          >
            View all {projects.length} projects{" "}
            <ArrowRight
              className="size-4 transition-transform group-hover:translate-x-1"
              aria-hidden="true"
            />
          </Link>
        </section>
      </BlurFade>

      <BlurFade delay={0.12}>
        <section
          id="about"
          aria-labelledby="about-heading"
          className="space-y-5"
        >
          <h2
            id="about-heading"
            className="text-xl font-semibold tracking-tight"
          >
            A little about me
          </h2>
          <p className="text-sm leading-7 text-muted-foreground">
            {profile.about}
          </p>
          <div className="flex items-start gap-3 rounded-xl border p-4">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-full bg-muted">
              <GraduationCap className="size-5" aria-hidden="true" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">
                {profile.education.school}
              </h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {profile.education.program}
              </p>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                {profile.education.focus}
              </p>
            </div>
          </div>
        </section>
      </BlurFade>

      <BlurFade delay={0.16}>
        <section aria-labelledby="toolkit-heading" className="space-y-4">
          <h2
            id="toolkit-heading"
            className="text-xl font-semibold tracking-tight"
          >
            What I work with
          </h2>
          <div className="flex flex-wrap gap-2">
            {profile.skills.map((skill) => (
              <Badge
                key={skill}
                variant="outline"
                className="rounded-lg px-3 py-1.5 text-xs font-normal"
              >
                {skill}
              </Badge>
            ))}
          </div>
        </section>
      </BlurFade>

      <BlurFade delay={0.2}>
        <section aria-labelledby="studies-heading" className="space-y-5">
          <div className="flex items-center justify-between gap-3">
            <h2
              id="studies-heading"
              className="text-xl font-semibold tracking-tight"
            >
              Notes from the learning process
            </h2>
            <BookOpen
              className="size-4 shrink-0 text-muted-foreground"
              aria-hidden="true"
            />
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            Implementations, experiments, and the ideas behind them.
          </p>
          <div className="divide-y">
            {studies.slice(0, 3).map((entry) => (
              <Link
                key={entry.slug}
                href={`/studies/${entry.slug}/`}
                className="group flex items-start justify-between gap-4 py-4"
              >
                <div>
                  <h3 className="text-sm font-medium group-hover:underline underline-offset-4">
                    {entry.title.replace(/^Online Learning - /, "")}
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                    {entry.tags.slice(0, 3).join(" · ")}
                  </p>
                </div>
                <ArrowUpRight
                  className="mt-0.5 size-4 shrink-0 text-muted-foreground"
                  aria-hidden="true"
                />
              </Link>
            ))}
          </div>
          <Link
            href="/studies/"
            className="inline-flex items-center gap-2 text-sm font-medium"
          >
            All {studies.length} studies{" "}
            <ArrowRight className="size-4" aria-hidden="true" />
          </Link>
        </section>
      </BlurFade>

      <BlurFade delay={0.24}>
        <section
          id="contact"
          aria-labelledby="contact-heading"
          className="rounded-2xl border bg-muted/30 px-6 py-8 text-center"
        >
          <h2
            id="contact-heading"
            className="text-2xl font-semibold tracking-tight"
          >
            Let’s build something thoughtful.
          </h2>
          <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-muted-foreground">
            For conversations about AI engineering, research, or a project you
            have in mind.
          </p>
          <a
            href={`mailto:${profile.email}`}
            className="mt-5 inline-flex items-center gap-2 text-sm font-medium underline underline-offset-4"
          >
            <Mail className="size-4" aria-hidden="true" />
            {profile.email}
          </a>
        </section>
      </BlurFade>
    </main>
  );
}
