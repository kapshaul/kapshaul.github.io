import type { Metadata } from "next";
import { notFound } from "next/navigation";
import EntryPage from "@/components/entry-page";
import { projectDisplay } from "@/data/projects";
import { getEntries, getEntry } from "@/lib/content";

export const dynamicParams = false;
export function generateStaticParams() {
  return getEntries("projects").map(({ slug }) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const entry = getEntry("projects", slug);
  if (!entry) return {};
  const display = projectDisplay[slug];
  const title = display?.title || entry.title;
  const description = display?.summary || entry.summary;
  return {
    title,
    description,
    alternates: { canonical: `/projects/${slug}/` },
    openGraph: {
      title,
      description,
      url: `/projects/${slug}/`,
      type: "article",
      ...(entry.date ? { publishedTime: entry.date } : {}),
      ...(entry.updated ? { modifiedTime: entry.updated } : {}),
    },
    twitter: { card: "summary", title, description },
  };
}

export default async function Project({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const entry = getEntry("projects", slug);
  if (!entry) notFound();
  return <EntryPage entry={entry} />;
}
