import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { featuredSlugs } from "@/data/projects";

export type Section = "projects" | "studies";

export type Entry = {
  slug: string;
  section: Section;
  title: string;
  summary: string;
  category: string | null;
  date: string | null;
  updated: string | null;
  tags: string[];
  authors: string[];
  image: string | null;
  body: string;
  links: { label: string; url: string }[];
};

const contentRoot = path.join(process.cwd(), "content");

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeLabel(value: unknown): string {
  return text(value)
    .replace(/Fine-Tunning/gi, "Fine-Tuning")
    .replace(/On Process/gi, "In Progress");
}

function strings(value: unknown): string[] {
  const values = Array.isArray(value) ? value : value ? [value] : [];
  return [...new Set(values.map(normalizeLabel).filter(Boolean))];
}

function date(value: unknown): string | null {
  if (!value) return null;
  const parsed = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(parsed.getTime()) || parsed.getUTCFullYear() < 1900)
    return null;
  return parsed.toISOString().slice(0, 10);
}

function resolveUrl(value: unknown, basePath: string): string | null {
  const raw = text(value);
  if (!raw) return null;
  try {
    const resolved = new URL(raw, `https://portfolio.invalid${basePath}`);
    if (!["https:", "http:", "mailto:"].includes(resolved.protocol))
      return null;
    return resolved.origin === "https://portfolio.invalid"
      ? `${resolved.pathname}${resolved.search}${resolved.hash}`
      : resolved.href;
  } catch {
    return null;
  }
}

function readEntry(
  section: Section,
  source: string,
  slug: string,
): Entry | null {
  const { data, content } = matter(fs.readFileSync(source, "utf8"));
  if (data.draft === true || data.draft === "true") return null;

  const basePath = `/${section}/${slug}/`;
  const links: Entry["links"] = [];
  const downloads =
    content.match(
      /^#{1,6}\s+Download\s*\r?\n([\s\S]*?)(?=^---\s*$|^#{1,6}\s|$(?![\s\S]))/m,
    )?.[1] ?? "";
  for (const match of downloads.matchAll(/\[([^\]]+)\]\(([^)]*)\)/g)) {
    const url = resolveUrl(match[2], basePath);
    if (url) links.push({ label: normalizeLabel(match[1]), url });
  }
  const repository = resolveUrl(data.editPost?.URL, basePath);
  if (
    repository?.startsWith("https://github.com/") &&
    !links.some((link) => link.url === repository)
  ) {
    links.push({ label: "Code", url: repository });
  }

  return {
    slug,
    section,
    title: normalizeLabel(data.title) || slug,
    summary: normalizeLabel(data.summary),
    category: normalizeLabel(data.category) || null,
    date: date(data.date),
    updated: date(data.lastmod),
    tags: strings(data.tags),
    authors: strings(data.author),
    image: resolveUrl(data.cover?.image, basePath),
    body: content,
    links,
  };
}

export function getEntries(section: Section): Entry[] {
  const directory = path.join(contentRoot, section);
  if (!fs.existsSync(directory)) return [];

  const entries = fs
    .readdirSync(directory, { withFileTypes: true })
    .flatMap((file) => {
      if (file.name.startsWith("_")) return [];
      const source = file.isDirectory()
        ? path.join(directory, file.name, "index.md")
        : path.join(directory, file.name);
      if (!source.endsWith(".md") || !fs.existsSync(source)) return [];
      const slug = file.isDirectory()
        ? file.name
        : file.name.replace(/\.md$/, "");
      const entry = readEntry(section, source, slug);
      return entry ? [entry] : [];
    });

  const byDate = (a: Entry, b: Entry) =>
    (b.date ?? "0000").localeCompare(a.date ?? "0000");
  if (section !== "projects") return entries.sort(byDate);

  // Projects follow the curated homepage order; the rest fall back to date.
  const rank = (entry: Entry) => {
    const index = featuredSlugs.indexOf(entry.slug);
    return index === -1 ? featuredSlugs.length : index;
  };
  return entries.sort((a, b) => rank(a) - rank(b) || byDate(a, b));
}

export function getEntry(section: Section, slug: string): Entry | undefined {
  return getEntries(section).find((entry) => entry.slug === slug);
}

// Client-side filtering only needs summaries, not entire technical articles.
export function getEntrySummaries(section: Section): Omit<Entry, "body">[] {
  return getEntries(section).map((entry) => {
    const { body, ...summary } = entry;
    void body;
    return summary;
  });
}
