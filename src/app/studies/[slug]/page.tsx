import type { Metadata } from "next";
import { notFound } from "next/navigation";
import EntryPage from "@/components/entry-page";
import { getEntries, getEntry } from "@/lib/content";

export const dynamicParams = false;
export function generateStaticParams() {
  return getEntries("studies").map(({ slug }) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const entry = getEntry("studies", slug);
  if (!entry) return {};
  return {
    title: entry.title,
    description: entry.summary,
    alternates: { canonical: `/studies/${slug}/` },
    openGraph: {
      title: entry.title,
      description: entry.summary,
      url: `/studies/${slug}/`,
      type: "article",
      ...(entry.date ? { publishedTime: entry.date } : {}),
    },
  };
}

export default async function Study({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const entry = getEntry("studies", slug);
  if (!entry) notFound();
  return <EntryPage entry={entry} />;
}
