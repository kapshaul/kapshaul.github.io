import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Navbar from "@/components/navbar";
import { FlickeringGrid } from "@/components/magicui/flickering-grid";
import { ThemeProvider } from "@/components/theme-provider";
import { TooltipProvider } from "@/components/ui/tooltip";
import { profile } from "@/data/profile";
import "katex/dist/katex.min.css";
import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist-sans" });
const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  metadataBase: new URL(profile.url),
  title: {
    default: `${profile.name} · AI Engineer`,
    template: `%s | ${profile.name}`,
  },
  description: profile.description,
  authors: [{ name: profile.name, url: profile.github }],
  openGraph: {
    title: `${profile.name} · AI Engineer`,
    description: profile.description,
    url: profile.url,
    siteName: profile.name,
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: `${profile.name} · AI Engineer`,
    description: profile.description,
  },
  robots: { index: true, follow: true },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geist.variable} ${geistMono.variable} min-h-screen bg-background font-sans text-foreground antialiased`}
      >
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          disableTransitionOnChange
        >
          <TooltipProvider delayDuration={150}>
            <FlickeringGrid />
            <div className="relative mx-auto max-w-[720px] px-5 pb-28 pt-16 sm:px-8 sm:pt-24">
              {children}
              <footer className="mt-20 flex flex-wrap items-center justify-between gap-3 border-t pt-6 text-[11px] text-muted-foreground">
                <span>
                  {profile.name} <span lang="ko">· {profile.koreanName}</span>
                </span>
                <a
                  href="https://github.com/magicuidesign/portfolio"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground"
                >
                  Built with Magic UI ↗
                </a>
              </footer>
            </div>
            <Navbar />
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
