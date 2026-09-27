/* eslint-disable @next/next/no-img-element */
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import rehypeKatex from "rehype-katex";

type HtmlNode = {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HtmlNode[];
};

// Keep the original figure groups without allowing arbitrary inline CSS.
// The same pass converts MathJax-style delimiters inside raw HTML captions.
function rehypeLegacyContent() {
  return (tree: HtmlNode) => {
    function walk(node: HtmlNode, inCode = false) {
      const classes = node.properties?.className;
      const classList = Array.isArray(classes) ? classes : [];
      const isMath = classList.some((value) =>
        ["language-math", "math-inline", "math-display"].includes(
          String(value),
        ),
      );
      if (isMath) {
        for (const child of node.children ?? []) {
          if (child.type === "text" && child.value) {
            child.value = child.value.replace(/\\\\(?=[a-zA-Z])/g, "\\");
          }
        }
      }

      if (node.tagName === "div" && node.properties) {
        const style = String(node.properties.style ?? "");
        if (/display\s*:\s*flex/i.test(style)) {
          node.properties.className = ["legacy-figure-grid"];
        } else if (
          /text-align\s*:\s*center/i.test(style) ||
          node.properties.align === "center"
        ) {
          node.properties.className = ["legacy-figure"];
        }
        delete node.properties.style;
      }

      const protectedText =
        inCode || node.tagName === "code" || node.tagName === "pre" || isMath;
      if (!node.children) return;
      node.children = node.children.flatMap((child) => {
        if (!protectedText && child.type === "text" && child.value) {
          const value = child.value;
          const matches = [
            ...value.matchAll(/\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g),
          ];
          if (matches.length) {
            const parts: HtmlNode[] = [];
            let cursor = 0;
            for (const match of matches) {
              const index = match.index ?? 0;
              if (index > cursor)
                parts.push({ type: "text", value: value.slice(cursor, index) });
              parts.push({
                type: "element",
                tagName: "span",
                properties: {
                  className: [
                    match[1] !== undefined ? "math-inline" : "math-display",
                  ],
                },
                children: [
                  { type: "text", value: (match[1] ?? match[2]).trim() },
                ],
              });
              cursor = index + match[0].length;
            }
            if (cursor < value.length)
              parts.push({ type: "text", value: value.slice(cursor) });
            return parts;
          }
        }
        walk(child, protectedText);
        return [child];
      });
    }
    walk(tree);
  };
}

const schema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    div: [
      ...(defaultSchema.attributes?.div ?? []),
      ["className", "legacy-figure", "legacy-figure-grid"],
    ],
    span: [
      ...(defaultSchema.attributes?.span ?? []),
      ["className", "math-inline", "math-display"],
    ],
    code: [
      ...(defaultSchema.attributes?.code ?? []),
      ["className", /^language-./, "math-inline", "math-display"],
    ],
    img: [...(defaultSchema.attributes?.img ?? []), "width", "height"],
  },
};

function normalizeLegacyMarkdown(content: string): string {
  let quotedMath = "";
  let codeFence = "";
  return content
    .split(/\r?\n/)
    .map((line) => {
      const fence = line.match(/^\s*(`{3,}|~{3,})/);
      if (fence) {
        codeFence = codeFence ? "" : fence[1][0];
        return line;
      }
      if (codeFence) return line;
      if (quotedMath) {
        const fixed = /^\s*>/.test(line) ? line : `${quotedMath}${line}`;
        if (/^\s*(?:>\s*)?\$\$\s*$/.test(line)) quotedMath = "";
        return fixed;
      }
      // Hugo tolerated quoted opening delimiters followed by unquoted formulas.
      // CommonMark needs every display-math line to remain in the same quote.
      if (/^\s*>\s*\$\$\s*$/.test(line)) quotedMath = "> ";
      return line;
    })
    .join("\n");
}

export function MarkdownContent({
  content,
  basePath,
}: {
  content: string;
  basePath: string;
}) {
  function resolveUrl(value: string): string {
    const safe = defaultUrlTransform(value);
    if (!safe) return "";
    try {
      const url = new URL(safe, `https://portfolio.invalid${basePath}`);
      if (!["https:", "http:", "mailto:"].includes(url.protocol)) return "";
      return url.origin === "https://portfolio.invalid"
        ? `${url.pathname}${url.search}${url.hash}`
        : url.href;
    } catch {
      return "";
    }
  }

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          rehypeRaw,
          rehypeLegacyContent,
          [rehypeSanitize, schema],
          [rehypeKatex, { strict: false }],
        ]}
        urlTransform={resolveUrl}
        components={{
          h1: ({ children }) => <h2>{children}</h2>,
          h5: ({ children }) => <h2>{children}</h2>,
          h6: ({ children }) => <h3>{children}</h3>,
          a: ({ href, children }) => {
            if (!href) return <span>{children}</span>;
            const external = /^https?:\/\//.test(href);
            return (
              <a
                href={href}
                target={external ? "_blank" : undefined}
                rel={external ? "noreferrer noopener" : undefined}
              >
                {children}
              </a>
            );
          },
          img: ({ src, alt, width, height }) => {
            if (!src)
              return (
                <span className="missing-figure">
                  Figure unavailable in the original archive.
                </span>
              );
            return (
              <img
                src={src}
                alt={alt || "Project illustration"}
                width={width}
                height={height}
                loading="lazy"
              />
            );
          },
          table: ({ children }) => (
            <div className="table-scroll">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {normalizeLegacyMarkdown(content)}
      </ReactMarkdown>
    </div>
  );
}
