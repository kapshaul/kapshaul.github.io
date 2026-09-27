import type { MetadataRoute } from "next";
import { getEntries } from "@/lib/content";
import { profile } from "@/data/profile";

export const dynamic = "force-static";
export default function sitemap(): MetadataRoute.Sitemap {
  return [
    ...["/", "/projects/", "/studies/"].map((route) => ({
      url: `${profile.url}${route}`,
    })),
    ...[...getEntries("projects"), ...getEntries("studies")].map((entry) => ({
      url: `${profile.url}/${entry.section}/${entry.slug}/`,
      ...(entry.updated || entry.date
        ? { lastModified: entry.updated || entry.date || undefined }
        : {}),
    })),
  ];
}
