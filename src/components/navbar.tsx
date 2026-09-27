"use client";

import {
  BookOpen,
  FolderKanban,
  Github,
  House,
  Mail,
  Moon,
  Sun,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { Dock, DockIcon } from "@/components/magicui/dock";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { profile } from "@/data/profile";

const links = [
  { href: "/", label: "Home", icon: House },
  { href: "/projects/", label: "Projects", icon: FolderKanban },
  { href: "/studies/", label: "Studies", icon: BookOpen },
  { href: profile.github, label: "GitHub", icon: Github },
  { href: `mailto:${profile.email}`, label: "Email", icon: Mail },
];

export default function Navbar() {
  const pathname = usePathname();
  const { resolvedTheme, setTheme } = useTheme();

  return (
    <nav
      aria-label="Main navigation"
      className="pointer-events-none fixed inset-x-0 bottom-4 z-40 px-3"
    >
      <Dock
        magnification={48}
        className="pointer-events-auto h-14 items-center gap-1.5 border bg-background/90 p-2 shadow-lg shadow-black/5 backdrop-blur-xl"
      >
        {links.map(({ href, label, icon: Icon }) => {
          const external = href.startsWith("https:");
          const active =
            href === "/"
              ? pathname === "/"
              : href.startsWith("/") && pathname.startsWith(href);
          return (
            <Tooltip key={href}>
              <TooltipTrigger asChild>
                <Link
                  href={href}
                  aria-label={label}
                  aria-current={active ? "page" : undefined}
                  target={external ? "_blank" : undefined}
                  rel={external ? "noopener noreferrer" : undefined}
                  className={`rounded-full transition-colors ${active ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}
                >
                  <DockIcon>
                    <Icon className="size-5" aria-hidden="true" />
                  </DockIcon>
                </Link>
              </TooltipTrigger>
              <TooltipContent sideOffset={12}>{label}</TooltipContent>
            </Tooltip>
          );
        })}
        <span aria-hidden="true" className="mx-0.5 h-6 w-px bg-border" />
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label="Toggle color theme"
              onClick={() =>
                setTheme(resolvedTheme === "dark" ? "light" : "dark")
              }
              className="rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <DockIcon>
                <Sun className="size-5 dark:hidden" aria-hidden="true" />
                <Moon className="hidden size-5 dark:block" aria-hidden="true" />
              </DockIcon>
            </button>
          </TooltipTrigger>
          <TooltipContent sideOffset={12}>Toggle theme</TooltipContent>
        </Tooltip>
      </Dock>
    </nav>
  );
}
