import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

const root = process.cwd();
const output = path.join(root, "out");
const failures = [];
const checked = new Set();
let routes = 0;

function checkLocalUrl(value, base, context) {
  if (!value || value.startsWith("#")) return;
  let url;
  try {
    url = new URL(value, `https://portfolio.invalid${base}`);
  } catch {
    return;
  }
  if (url.origin !== "https://portfolio.invalid") return;
  const pathname = decodeURIComponent(url.pathname);
  const key = `${context}: ${pathname}`;
  if (checked.has(key)) return;
  checked.add(key);
  const target = path.join(output, pathname);
  const candidates = [
    target,
    `${target}.html`,
    path.join(target, "index.html"),
  ];
  if (
    !candidates.some(
      (candidate) =>
        fs.existsSync(candidate) && fs.statSync(candidate).isFile(),
    )
  ) {
    failures.push(`${context}: missing ${pathname}`);
  }
}

function checkRenderedPage(filename, route) {
  const html = fs.readFileSync(filename, "utf8");
  for (const match of html.matchAll(/\b(?:href|src)="([^"#]+)"/g)) {
    checkLocalUrl(match[1].replaceAll("&amp;", "&"), route, route);
  }
  if (html.includes('class="katex-error"'))
    failures.push(`${route}: invalid math expression`);
}

if (!fs.existsSync(output)) {
  console.error("Static export is missing. Run the production build first.");
  process.exit(1);
}

if (fs.existsSync(path.join(output, "static")))
  failures.push("Duplicate static/ directory unexpectedly exported");

for (const section of ["projects", "studies"]) {
  const directory = path.join(root, "content", section);
  for (const item of fs.readdirSync(directory, { withFileTypes: true })) {
    if (item.name.startsWith("_")) continue;
    const source = item.isDirectory()
      ? path.join(directory, item.name, "index.md")
      : path.join(directory, item.name);
    if (!source.endsWith(".md") || !fs.existsSync(source)) continue;
    const { data } = matter(fs.readFileSync(source, "utf8"));
    const slug = item.isDirectory()
      ? item.name
      : item.name.replace(/\.md$/, "");
    const route = `/${section}/${slug}/`;
    const file = path.join(output, section, slug, "index.html");
    if (data.draft === true || data.draft === "true") {
      if (fs.existsSync(file))
        failures.push(`Draft article unexpectedly exported: ${route}`);
      continue;
    }
    routes += 1;
    if (!fs.existsSync(file)) failures.push(`Missing article route: ${route}`);
    else checkRenderedPage(file, route);
  }
}

for (const route of ["/", "/projects/", "/studies/"]) {
  const file = path.join(output, route, "index.html");
  if (!fs.existsSync(file)) failures.push(`Missing index route: ${route}`);
  else checkRenderedPage(file, route);
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(
  `Verified ${routes} published articles, 3 index pages, and ${checked.size} local links/assets; drafts excluded.`,
);
